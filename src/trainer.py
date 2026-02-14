"""
ModernBERT-RGAT | Trainer
==========================
Full training engine for the joint ATE + ASC model.

Features:
  - Mixed precision training (FP16)
  - Gradient clipping
  - Linear warmup + decay scheduling
  - Early stopping on validation F1
  - Strict F1 for ATE (exact span matching)
  - Checkpoint save/load
  - Per-epoch logging with training curves
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup

from src.losses import JointTaskLoss


# ---------------------------------------------------------------------------
#  BIO Span Extraction (for Strict F1)
# ---------------------------------------------------------------------------

def extract_spans_from_bio(bio_sequence: List[int]) -> List[Tuple[int, int]]:
    """
    Extract aspect spans from a BIO tag sequence.

    Decodes a list of BIO tag indices into (start, end) span tuples.
    Uses strict BIO decoding: a span starts with B-ASP (1) and continues
    through consecutive I-ASP (2) tags.

    Args:
        bio_sequence: list of ints (0=O, 1=B-ASP, 2=I-ASP, -100=ignore)

    Returns:
        List of (start_idx, end_idx) tuples (end is exclusive)
    """
    spans = []
    current_start = None

    for i, tag in enumerate(bio_sequence):
        if tag == 1:  # B-ASP
            # Close previous span if open
            if current_start is not None:
                spans.append((current_start, i))
            current_start = i
        elif tag == 2:  # I-ASP
            # Continue current span (only valid after B-ASP)
            if current_start is None:
                # I-ASP without preceding B-ASP -- treat as B-ASP
                current_start = i
        else:
            # O or ignore: close current span if open
            if current_start is not None:
                spans.append((current_start, i))
                current_start = None

    # Close final span if still open
    if current_start is not None:
        spans.append((current_start, len(bio_sequence)))

    return spans


def compute_strict_f1(
    pred_bio_batch: List[List[int]],
    gold_bio_batch: List[List[int]],
) -> Dict[str, float]:
    """
    Compute strict F1 for ATE across a batch.

    Strict matching: a predicted span is correct only if both its start
    and end indices match a gold span exactly.

    Args:
        pred_bio_batch: list of predicted BIO sequences
        gold_bio_batch: list of gold BIO sequences

    Returns:
        dict with precision, recall, f1
    """
    total_pred = 0
    total_gold = 0
    total_correct = 0

    for pred_seq, gold_seq in zip(pred_bio_batch, gold_bio_batch):
        pred_spans = set(extract_spans_from_bio(pred_seq))
        gold_spans = set(extract_spans_from_bio(gold_seq))

        total_pred += len(pred_spans)
        total_gold += len(gold_spans)
        total_correct += len(pred_spans & gold_spans)

    precision = total_correct / max(total_pred, 1)
    recall = total_correct / max(total_gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {"precision": precision, "recall": recall, "f1": f1}


# ---------------------------------------------------------------------------
#  Training Metrics Tracker
# ---------------------------------------------------------------------------

class MetricsTracker:
    """
    Tracks training and validation metrics across epochs for logging
    and visualization.
    """

    def __init__(self):
        self.history = {
            "train_loss": [],
            "train_ate_loss": [],
            "train_asc_loss": [],
            "val_loss": [],
            "val_ate_loss": [],
            "val_asc_loss": [],
            "val_ate_f1": [],
            "val_asc_accuracy": [],
            "val_asc_f1": [],
            "learning_rate": [],
            "alpha": [],
        }

    def log_train(self, loss: float, ate_loss: float, asc_loss: float,
                  lr: float, alpha: float):
        self.history["train_loss"].append(loss)
        self.history["train_ate_loss"].append(ate_loss)
        self.history["train_asc_loss"].append(asc_loss)
        self.history["learning_rate"].append(lr)
        self.history["alpha"].append(alpha)

    def log_val(self, loss: float, ate_loss: float, asc_loss: float,
                ate_f1: float, asc_accuracy: float, asc_f1: float):
        self.history["val_loss"].append(loss)
        self.history["val_ate_loss"].append(ate_loss)
        self.history["val_asc_loss"].append(asc_loss)
        self.history["val_ate_f1"].append(ate_f1)
        self.history["val_asc_accuracy"].append(asc_accuracy)
        self.history["val_asc_f1"].append(asc_f1)

    def get_history(self) -> Dict:
        return self.history


# ---------------------------------------------------------------------------
#  Early Stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """
    Stops training if the monitored metric doesn't improve for `patience`
    consecutive epochs.

    Args:
        patience: number of epochs to wait before stopping
        mode:     'max' (higher is better, e.g. F1) or 'min' (lower is better)
        delta:    minimum change to count as improvement
    """

    def __init__(self, patience: int = 5, mode: str = "max", delta: float = 0.0):
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.

        Args:
            score: current metric value

        Returns:
            True if this is the best score so far (should save checkpoint)
        """
        if self.best_score is None:
            self.best_score = score
            return True  # first epoch is always "best"

        if self.mode == "max":
            improved = score > self.best_score + self.delta
        else:
            improved = score < self.best_score - self.delta

        if improved:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return False


# ---------------------------------------------------------------------------
#  Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """
    Training engine for the ModernBERT-RGAT joint model.

    Handles the full training lifecycle: forward pass, loss computation,
    backpropagation, validation, checkpointing, and early stopping.

    Args:
        model:           ModernBERT_RGAT model instance
        train_loader:    training DataLoader
        val_loader:      validation DataLoader
        config:          full config dict (from config.yaml)
        sentiment_weights: class weight tensor for ASC focal loss
        device:          torch device ('cuda' or 'cpu')
        dataset_year:    which dataset year (for checkpoint naming)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        sentiment_weights: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dataset_year: str = "2014",
    ):
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_year = dataset_year

        # Move model to device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training config
        train_cfg = config["training"]
        self.epochs = train_cfg["epochs"]
        self.max_grad_norm = train_cfg["max_grad_norm"]
        self.use_fp16 = train_cfg.get("fp16", False) and torch.cuda.is_available()
        self.log_every_n_steps = config.get("logging", {}).get("log_every_n_steps", 50)

        # Total training steps (for scheduler and alpha scheduling)
        self.total_steps = len(train_loader) * self.epochs

        # Optimizer: AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )

        # LR Scheduler: linear warmup then linear decay
        warmup_steps = int(self.total_steps * train_cfg["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps,
        )

        # Joint loss function with dynamic alpha and focal loss
        alpha_start = train_cfg.get("alpha_start", 0.7)
        alpha_end = train_cfg.get("alpha_end", 0.3)
        focal_gamma = train_cfg.get("focal_gamma", 2.0)

        if sentiment_weights is not None:
            sentiment_weights = sentiment_weights.to(self.device)

        self.loss_fn = JointTaskLoss(
            sentiment_weights=sentiment_weights,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            gamma=focal_gamma,
            total_steps=self.total_steps,
        )

        # Mixed precision scaler
        self.scaler = GradScaler("cuda", enabled=self.use_fp16)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 5),
            mode=es_cfg.get("mode", "max"),
        )

        # Metrics tracking
        self.metrics = MetricsTracker()

        # Checkpoint directory
        self.checkpoint_dir = train_cfg.get("checkpoint_dir", "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Print training setup summary
        print(f"\n{'='*60}")
        print(f"  Trainer initialized for SemEval {dataset_year}")
        print(f"{'='*60}")
        print(f"  Device:          {self.device}")
        print(f"  Epochs:          {self.epochs}")
        print(f"  Total steps:     {self.total_steps}")
        print(f"  Warmup steps:    {warmup_steps}")
        print(f"  Learning rate:   {train_cfg['learning_rate']}")
        print(f"  Batch size:      {train_cfg['batch_size']}")
        print(f"  FP16:            {self.use_fp16}")
        print(f"  Alpha schedule:  {alpha_start} -> {alpha_end}")
        print(f"  Focal gamma:     {focal_gamma}")
        print(f"  Early stopping:  patience={es_cfg.get('patience', 5)}")
        print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    #  Training Loop (one epoch)
    # ------------------------------------------------------------------

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns dict with average loss values for this epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_ate_loss = 0.0
        total_asc_loss = 0.0
        num_batches = 0

        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            adj_matrix = batch["adj_matrix"].to(self.device)
            aspect_mask = batch["aspect_mask"].to(self.device)
            bio_labels = batch["bio_labels"].to(self.device)
            sentiment_labels = batch["sentiment_label"].to(self.device)

            # Forward pass with mixed precision
            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    adj_matrix=adj_matrix,
                    aspect_mask=aspect_mask,
                )
                loss, ate_loss, asc_loss = self.loss_fn(
                    ate_logits=outputs["ate_logits"],
                    ate_labels=bio_labels,
                    asc_logits=outputs["sentiment_logits"],
                    asc_labels=sentiment_labels,
                )

            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm computation)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.loss_fn.step()  # advance dynamic alpha

            # Accumulate metrics
            total_loss += loss.item()
            total_ate_loss += ate_loss.item()
            total_asc_loss += asc_loss.item()
            num_batches += 1

            # Step-level logging
            global_step = epoch * len(self.train_loader) + step
            if (step + 1) % self.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                current_lr = self.scheduler.get_last_lr()[0]
                current_alpha = self.loss_fn.current_alpha
                print(
                    f"  Step {step+1}/{len(self.train_loader)} | "
                    f"loss={avg_loss:.4f} | "
                    f"ate={total_ate_loss/num_batches:.4f} | "
                    f"asc={total_asc_loss/num_batches:.4f} | "
                    f"lr={current_lr:.2e} | "
                    f"alpha={current_alpha:.3f}"
                )

        # Epoch averages
        avg_loss = total_loss / max(num_batches, 1)
        avg_ate = total_ate_loss / max(num_batches, 1)
        avg_asc = total_asc_loss / max(num_batches, 1)
        current_lr = self.scheduler.get_last_lr()[0]
        current_alpha = self.loss_fn.current_alpha

        self.metrics.log_train(avg_loss, avg_ate, avg_asc, current_lr, current_alpha)

        return {
            "loss": avg_loss,
            "ate_loss": avg_ate,
            "asc_loss": avg_asc,
        }

    # ------------------------------------------------------------------
    #  Validation Loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """
        Run validation and compute all metrics.

        Returns dict with loss values and evaluation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_ate_loss = 0.0
        total_asc_loss = 0.0
        num_batches = 0

        all_ate_preds = []
        all_ate_labels = []
        all_asc_preds = []
        all_asc_labels = []

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            adj_matrix = batch["adj_matrix"].to(self.device)
            aspect_mask = batch["aspect_mask"].to(self.device)
            bio_labels = batch["bio_labels"].to(self.device)
            sentiment_labels = batch["sentiment_label"].to(self.device)

            with autocast(device_type=self.device.type, enabled=self.use_fp16):
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    adj_matrix=adj_matrix,
                    aspect_mask=aspect_mask,
                )
                loss, ate_loss, asc_loss = self.loss_fn(
                    ate_logits=outputs["ate_logits"],
                    ate_labels=bio_labels,
                    asc_logits=outputs["sentiment_logits"],
                    asc_labels=sentiment_labels,
                )

            total_loss += loss.item()
            total_ate_loss += ate_loss.item()
            total_asc_loss += asc_loss.item()
            num_batches += 1

            # Collect predictions for metrics
            ate_preds = outputs["ate_logits"].argmax(dim=-1)  # [batch, seq_len]
            asc_preds = outputs["sentiment_logits"].argmax(dim=-1)  # [batch]

            # Convert to lists, filtering out -100 for ATE
            for i in range(ate_preds.size(0)):
                pred_seq = ate_preds[i].cpu().tolist()
                gold_seq = bio_labels[i].cpu().tolist()

                # For strict F1, only keep positions where gold != -100
                filtered_pred = []
                filtered_gold = []
                for p, g in zip(pred_seq, gold_seq):
                    if g != -100:
                        filtered_pred.append(p)
                        filtered_gold.append(g)

                all_ate_preds.append(filtered_pred)
                all_ate_labels.append(filtered_gold)

            all_asc_preds.extend(asc_preds.cpu().tolist())
            all_asc_labels.extend(sentiment_labels.cpu().tolist())

        # Compute ATE Strict F1
        ate_metrics = compute_strict_f1(all_ate_preds, all_ate_labels)

        # Compute ASC metrics
        asc_preds_arr = np.array(all_asc_preds)
        asc_labels_arr = np.array(all_asc_labels)
        asc_accuracy = (asc_preds_arr == asc_labels_arr).mean()

        # Per-class F1 for ASC (macro average)
        from sklearn.metrics import f1_score
        asc_f1 = f1_score(asc_labels_arr, asc_preds_arr, average="macro", zero_division=0)

        # Log validation metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_ate = total_ate_loss / max(num_batches, 1)
        avg_asc = total_asc_loss / max(num_batches, 1)

        self.metrics.log_val(
            avg_loss, avg_ate, avg_asc,
            ate_metrics["f1"], asc_accuracy, asc_f1,
        )

        return {
            "loss": avg_loss,
            "ate_loss": avg_ate,
            "asc_loss": avg_asc,
            "ate_precision": ate_metrics["precision"],
            "ate_recall": ate_metrics["recall"],
            "ate_f1": ate_metrics["f1"],
            "asc_accuracy": asc_accuracy,
            "asc_f1": asc_f1,
        }

    # ------------------------------------------------------------------
    #  Checkpoint Management
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, val_metrics: Dict):
        """Save model checkpoint with metadata."""
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"best_model_{self.dataset_year}.pt"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_metrics": val_metrics,
            "config": self.config,
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"  Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(self, path: str):
        """Load a previously saved checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"  Loaded checkpoint from {path} (epoch {checkpoint['epoch']})")
        return checkpoint

    # ------------------------------------------------------------------
    #  Full Training Run
    # ------------------------------------------------------------------

    def train(self) -> Dict:
        """
        Run the full training loop across all epochs.

        Returns:
            Dict with final metrics and training history.
        """
        print(f"\nStarting training for SemEval {self.dataset_year}...")
        print(f"{'='*60}\n")

        best_val_metrics = None
        start_time = time.time()

        for epoch in range(self.epochs):
            epoch_start = time.time()

            # --- Train ---
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_metrics = self.train_epoch(epoch)
            epoch_time = time.time() - epoch_start

            # --- Validate ---
            val_metrics = self.validate()

            # --- Print epoch summary ---
            print(
                f"\n  Train | loss={train_metrics['loss']:.4f} "
                f"ate={train_metrics['ate_loss']:.4f} "
                f"asc={train_metrics['asc_loss']:.4f}"
            )
            print(
                f"  Val   | loss={val_metrics['loss']:.4f} "
                f"ate={val_metrics['ate_loss']:.4f} "
                f"asc={val_metrics['asc_loss']:.4f}"
            )
            print(
                f"  ATE   | P={val_metrics['ate_precision']:.4f} "
                f"R={val_metrics['ate_recall']:.4f} "
                f"F1={val_metrics['ate_f1']:.4f}"
            )
            print(
                f"  ASC   | Acc={val_metrics['asc_accuracy']:.4f} "
                f"F1={val_metrics['asc_f1']:.4f}"
            )
            print(f"  Time  | {epoch_time:.1f}s | "
                  f"Alpha={self.loss_fn.current_alpha:.3f}")

            # --- Early stopping check ---
            # Using a combined metric: average of ATE F1 and ASC F1
            combined_f1 = (val_metrics["ate_f1"] + val_metrics["asc_f1"]) / 2
            is_best = self.early_stopping(combined_f1)

            if is_best:
                print(f"  >> New best combined F1: {combined_f1:.4f} -- saving checkpoint")
                self.save_checkpoint(epoch + 1, val_metrics)
                best_val_metrics = val_metrics

            if self.early_stopping.should_stop:
                print(f"\n  Early stopping triggered at epoch {epoch+1} "
                      f"(no improvement for {self.early_stopping.patience} epochs)")
                break

            print(f"  Patience: {self.early_stopping.counter}/{self.early_stopping.patience}")
            print(f"{'-'*60}")

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Training complete for SemEval {self.dataset_year}")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best combined F1: {self.early_stopping.best_score:.4f}")
        print(f"{'='*60}\n")

        # Save training history
        history_path = os.path.join(
            self.checkpoint_dir, f"history_{self.dataset_year}.json"
        )
        with open(history_path, "w") as f:
            json.dump(self.metrics.get_history(), f, indent=2)
        print(f"  Training history saved: {history_path}")

        return {
            "best_val_metrics": best_val_metrics,
            "history": self.metrics.get_history(),
            "total_time_seconds": total_time,
        }
