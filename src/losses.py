"""
ModernBERT-RGAT | Joint Task Loss
==================================
Combined loss function for joint Aspect Term Extraction (ATE) and
Aspect Sentiment Classification (ASC).

Key features:
  - ATE: standard CrossEntropyLoss with ignore_index=-100 for sub-tokens
  - ASC: Focal Loss with class weights for handling imbalanced sentiments
  - Dynamic alpha scheduling: starts favoring ATE (alpha=0.7), linearly
    shifts to favor ASC (alpha=0.3) as training progresses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
#  Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Applies a modulating factor (1 - p_t)^gamma to the standard cross-entropy
    loss, down-weighting easy examples and focusing on hard ones. This is
    particularly useful for the ASC task where the "conflict" and "neutral"
    classes are both rare and harder to distinguish.

    Reference: Lin et al., "Focal Loss for Dense Object Detection" (2017)

    Args:
        gamma:       focusing parameter. gamma=0 reduces to standard CE.
                     Higher gamma = more focus on hard examples.
        weight:      class weights tensor of shape [num_classes]
        reduction:   'mean', 'sum', or 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Register weight as buffer so it moves with .to(device) automatically
        if weight is not None:
            self.register_buffer("weight", weight)
        else:
            self.weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits:  [batch, num_classes] raw model output
            targets: [batch] integer class labels

        Returns:
            Scalar loss value
        """
        # Standard cross-entropy per sample (no reduction)
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.weight, reduction="none"
        )

        # Get predicted probability for the true class
        p_t = torch.exp(-ce_loss)

        # Apply the focal modulating factor
        focal_factor = (1.0 - p_t) ** self.gamma
        focal_loss = focal_factor * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
#  Dynamic Alpha Scheduler
# ---------------------------------------------------------------------------

class AlphaScheduler:
    """
    Linearly interpolates alpha from alpha_start to alpha_end over
    total_steps training steps.

    alpha controls the ATE vs ASC loss balance:
      total_loss = alpha * ate_loss + (1 - alpha) * asc_loss

    Starting with alpha=0.7 gives more weight to ATE early on, helping
    the model learn aspect boundaries first. Gradually shifting to
    alpha=0.3 lets the model focus on sentiment refinement once aspects
    are found reliably.

    Args:
        alpha_start: initial alpha value (high = favor ATE)
        alpha_end:   final alpha value (low = favor ASC)
        total_steps: total training steps for the schedule
    """

    def __init__(
        self,
        alpha_start: float = 0.7,
        alpha_end: float = 0.3,
        total_steps: int = 1000,
    ):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.total_steps = max(total_steps, 1)
        self.current_step = 0

    def get_alpha(self) -> float:
        """Return the current alpha value based on progress."""
        progress = min(self.current_step / self.total_steps, 1.0)
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * progress
        return alpha

    def step(self):
        """Advance the scheduler by one step."""
        self.current_step += 1


# ---------------------------------------------------------------------------
#  Joint Task Loss
# ---------------------------------------------------------------------------

class JointTaskLoss(nn.Module):
    """
    Combined loss for joint ATE + ASC training.

    Components:
      - ATE loss: CrossEntropyLoss with ignore_index=-100
        (skips sub-tokens, special tokens, and padding automatically)
      - ASC loss: FocalLoss with class weights and gamma focusing
      - Dynamic alpha: linearly shifts balance from ATE to ASC

    Args:
        sentiment_weights: class weight tensor for ASC [num_classes]
        alpha_start:       initial alpha (ATE weight), default 0.7
        alpha_end:         final alpha (ATE weight), default 0.3
        gamma:             focal loss gamma, default 2.0
        total_steps:       total training steps for alpha scheduling
        num_bio_tags:      number of BIO tags (default 3: O, B-ASP, I-ASP)
    """

    def __init__(
        self,
        sentiment_weights: Optional[torch.Tensor] = None,
        alpha_start: float = 0.7,
        alpha_end: float = 0.3,
        gamma: float = 2.0,
        total_steps: int = 1000,
        num_bio_tags: int = 3,
    ):
        super().__init__()
        self.num_bio_tags = num_bio_tags

        # ATE: standard cross-entropy that ignores -100 labels
        self.ate_criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # ASC: focal loss with class weights for imbalanced sentiments
        self.asc_criterion = FocalLoss(
            gamma=gamma, weight=sentiment_weights
        )

        # Dynamic alpha scheduler
        self.alpha_scheduler = AlphaScheduler(
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            total_steps=total_steps,
        )

    @property
    def current_alpha(self) -> float:
        """Current ATE-ASC balance weight."""
        return self.alpha_scheduler.get_alpha()

    def forward(
        self,
        ate_logits: torch.Tensor,
        ate_labels: torch.Tensor,
        asc_logits: torch.Tensor,
        asc_labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the joint loss.

        Args:
            ate_logits: [batch, seq_len, num_bio_tags] — ATE predictions
            ate_labels: [batch, seq_len] — BIO ground truth (-100 for ignored)
            asc_logits: [batch, num_sentiment_classes] — ASC predictions
            asc_labels: [batch] — sentiment ground truth (0-3)

        Returns:
            (total_loss, ate_loss, asc_loss) — all scalar tensors
        """
        # ATE loss: reshape to [batch*seq_len, num_tags] vs [batch*seq_len]
        ate_loss = self.ate_criterion(
            ate_logits.view(-1, self.num_bio_tags),
            ate_labels.view(-1),
        )

        # ASC loss: focal loss with class weights
        asc_loss = self.asc_criterion(asc_logits, asc_labels)

        # Dynamic weighted combination
        alpha = self.current_alpha
        total_loss = alpha * ate_loss + (1.0 - alpha) * asc_loss

        return total_loss, ate_loss, asc_loss

    def step(self):
        """Advance the alpha scheduler by one step. Call after each optimizer step."""
        self.alpha_scheduler.step()
