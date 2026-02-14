"""
ModernBERT-RGAT | Joint Model Architecture
============================================
Joint Aspect Term Extraction (ATE) + Aspect Sentiment Classification (ASC)

Architecture:
    Input -> ModernBERT -> RGAT (learnable relation weights) ->
        |-- ATE Head   (token-level BIO tagger: B-ASP / I-ASP / O)
        |-- ASC Head   (CLS + MaxPool(aspect_span) -> polarity)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ------------------------------------------------------------------
#  Relational Graph Attention Layer (with Learnable Relation Weights)
# ------------------------------------------------------------------

class RGATLayer(nn.Module):
    """
    Relational Graph Attention Layer.

    Refines token embeddings using syntactic dependency relations.
    Each of the 7 relation types gets:
      - Its own linear projection
      - A learnable importance weight (sigmoid-bounded)
      - A gated skip-connection to preserve BERT knowledge

    Relations:
        0: nsubj    (nominal subject)
        1: amod     (adjective modifier)   <-- high sentiment signal
        2: obj      (direct object)
        3: advmod   (adverb modifier)      <-- high sentiment signal
        4: neg      (negation)             <-- critical for sentiment
        5: compound (multi-word concepts)
        6: conj     (conjunctions)
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int = 7, dropout: float = 0.1):
        super().__init__()
        self.num_relations = num_relations
        self.out_dim = out_dim

        # Per-relation linear projections
        self.relation_projections = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False)
            for _ in range(num_relations)
        ])

        # Learnable relation importance weights -
        # the model learns which dependency types matter most for sentiment
        self.relation_importance = nn.Parameter(torch.ones(num_relations))

        # Attention mechanism for neighbor aggregation
        self.attn_linear = nn.Linear(out_dim * 2, 1)

        # Gated skip-connection to preserve original BERT embeddings
        self.gate_linear = nn.Linear(in_dim + out_dim, out_dim)
        self.gate_sigmoid = nn.Linear(in_dim + out_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:   Token embeddings from BERT  [batch, seq_len, in_dim]
            adj: Adjacency matrices          [batch, num_relations, seq_len, seq_len]

        Returns:
            Refined embeddings               [batch, seq_len, out_dim]
        """
        batch_size, seq_len, _ = h.size()
        relation_outputs = []

        for i in range(self.num_relations):
            # Project tokens for this relation type
            rel_h = self.relation_projections[i](h)          # [batch, seq, out_dim]

            # Aggregate neighbors via adjacency matrix
            context = torch.matmul(adj[:, i, :, :], rel_h)   # [batch, seq, out_dim]

            # Scale by learnable importance (sigmoid bounds to [0, 1])
            importance = torch.sigmoid(self.relation_importance[i])
            context = importance * context

            relation_outputs.append(context)

        # Sum across all relation types
        combined = torch.stack(relation_outputs, dim=0).sum(dim=0)  # [batch, seq, out_dim]
        combined = self.dropout(combined)

        # Gated skip-connection
        gate_input = torch.cat([h, combined], dim=-1)          # [batch, seq, in_dim + out_dim]
        gate = torch.sigmoid(self.gate_sigmoid(gate_input))    # [batch, seq, out_dim]
        transform = F.relu(self.gate_linear(gate_input))       # [batch, seq, out_dim]

        output = gate * transform + (1 - gate) * combined
        output = self.layer_norm(output)

        return output


# ------------------------------------------------------------------
#  Aspect Term Extraction Head (BIO Tagger)
# ------------------------------------------------------------------

class ATEHead(nn.Module):
    """
    Token-level BIO sequence labeler for Aspect Term Extraction.

    Each token gets classified as:
        O     (0): Outside - not part of any aspect
        B-ASP (1): Beginning of an aspect term
        I-ASP (2): Inside/continuation of an aspect term

    Sub-word alignment: non-first sub-tokens receive label -100
    (handled in dataset.py, ignored by CrossEntropyLoss).
    """

    def __init__(self, hidden_dim: int, num_tags: int = 3, dropout: float = 0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_tags),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_embeddings: [batch, seq_len, hidden_dim]

        Returns:
            BIO logits: [batch, seq_len, num_tags]
        """
        return self.classifier(token_embeddings)


# ------------------------------------------------------------------
#  Aspect Sentiment Classification Head (CLS + MaxPool)
# ------------------------------------------------------------------

class ASCHead(nn.Module):
    """
    Aspect-level sentiment classifier.

    Input strategy:
        sentiment_input = [CLS_embedding ; MaxPool(aspect_span_embeddings)]

    This concatenates:
        - CLS token (global sentence context)
        - Max-pooled aspect span (local aspect-specific features)

    For implicit aspects ([ASPECT], no span), CLS is duplicated:
        sentiment_input = [CLS ; CLS]
    """

    def __init__(self, hidden_dim: int, num_classes: int = 4, dropout: float = 0.1):
        super().__init__()
        # Input is [CLS ; MaxPool(span)] -> 2 * hidden_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        cls_embedding: torch.Tensor,
        token_embeddings: torch.Tensor,
        aspect_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cls_embedding:    [batch, hidden_dim]   - CLS token embedding
            token_embeddings: [batch, seq_len, hidden_dim]  - all token embeddings
            aspect_masks:     [batch, seq_len]  - 1 where aspect tokens are, 0 elsewhere

        Returns:
            Sentiment logits: [batch, num_classes]
        """
        # Max-pool over aspect span tokens
        # Where mask is 0, set embeddings to -inf so they don't affect max
        mask_expanded = aspect_masks.unsqueeze(-1)  # [batch, seq_len, 1]

        # Check if any aspect tokens exist per sample
        has_aspect = aspect_masks.sum(dim=1, keepdim=True) > 0  # [batch, 1]

        # Masked embeddings: set non-aspect positions to -inf for max pooling
        masked_embeddings = token_embeddings.masked_fill(
            mask_expanded == 0, float('-inf')
        )
        aspect_pooled = masked_embeddings.max(dim=1).values  # [batch, hidden_dim]

        # For samples with no aspect tokens (implicit), use CLS instead
        has_aspect_expanded = has_aspect.expand_as(cls_embedding)  # [batch, hidden_dim]
        aspect_pooled = torch.where(has_aspect_expanded, aspect_pooled, cls_embedding)

        # Concatenate: [CLS ; MaxPool(aspect_span)]
        sentiment_input = torch.cat([cls_embedding, aspect_pooled], dim=-1)  # [batch, 2*hidden_dim]

        return self.classifier(sentiment_input)


# ------------------------------------------------------------------
#  Full Joint Model: ModernBERT-RGAT
# ------------------------------------------------------------------

class ModernBERT_RGAT(nn.Module):
    """
    Joint Aspect Term Extraction + Aspect Sentiment Classification.

    Architecture:
        ModernBERT (semantic backbone)
            -> RGAT Layer (syntactic structure, learnable relation weights)
                -> ATE Head  (token-level BIO tagging)
                -> ASC Head  (CLS + MaxPool -> sentiment classification)

    Forward outputs:
        - ate_logits:       [batch, seq_len, 3]   - BIO tag predictions
        - sentiment_logits: [batch, 4]             - polarity predictions
    """

    def __init__(
        self,
        model_name: str = "answerdotai/ModernBERT-base",
        hidden_dim: int = 768,
        num_sentiment_classes: int = 4,
        num_bio_tags: int = 3,
        num_relations: int = 7,
        rgat_dropout: float = 0.1,
        head_dropout: float = 0.1,
    ):
        super().__init__()

        # 1. Semantic backbone
        self.bert = AutoModel.from_pretrained(model_name)

        # 2. Syntactic layer (RGAT)
        self.rgat = RGATLayer(
            in_dim=hidden_dim,
            out_dim=hidden_dim,
            num_relations=num_relations,
            dropout=rgat_dropout,
        )

        # 3. Task heads
        self.ate_head = ATEHead(
            hidden_dim=hidden_dim,
            num_tags=num_bio_tags,
            dropout=head_dropout,
        )
        self.asc_head = ASCHead(
            hidden_dim=hidden_dim,
            num_classes=num_sentiment_classes,
            dropout=head_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        adj_matrix: torch.Tensor,
        aspect_mask: torch.Tensor = None,
    ) -> dict:
        """
        Joint forward pass.

        Args:
            input_ids:      [batch, seq_len]
            attention_mask: [batch, seq_len]
            adj_matrix:     [batch, num_relations, seq_len, seq_len]
            aspect_mask:    [batch, seq_len]  - 1 at aspect token positions.
                            If None (inference), derived from ATE predictions.

        Returns:
            dict with:
                'ate_logits':       [batch, seq_len, num_bio_tags]
                'sentiment_logits': [batch, num_sentiment_classes]
                'graph_embeddings': [batch, seq_len, hidden_dim]
        """
        # A. ModernBERT contextual embeddings
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        sequence_output = bert_output.last_hidden_state  # [batch, seq_len, 768]

        # B. RGAT: inject syntactic structure
        graph_output = self.rgat(sequence_output, adj_matrix)  # [batch, seq_len, 768]

        # C. ATE Head: BIO tagging on every token
        ate_logits = self.ate_head(graph_output)  # [batch, seq_len, 3]

        # D. ASC Head: sentiment classification
        cls_embedding = graph_output[:, 0, :]  # [batch, 768] - CLS token

        if aspect_mask is None:
            # Inference mode: derive aspect mask from ATE predictions
            ate_preds = ate_logits.argmax(dim=-1)   # [batch, seq_len]
            aspect_mask = (ate_preds > 0).float()   # B-ASP=1, I-ASP=2 -> both > 0

        sentiment_logits = self.asc_head(
            cls_embedding=cls_embedding,
            token_embeddings=graph_output,
            aspect_masks=aspect_mask,
        )

        return {
            'ate_logits': ate_logits,
            'sentiment_logits': sentiment_logits,
            'graph_embeddings': graph_output,
        }

    def get_param_count(self) -> dict:
        """Get parameter counts by component."""
        def _count(module):
            return sum(p.numel() for p in module.parameters())

        return {
            'bert': _count(self.bert),
            'rgat': _count(self.rgat),
            'ate_head': _count(self.ate_head),
            'asc_head': _count(self.asc_head),
            'total': _count(self),
            'trainable': sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def get_relation_importance(self) -> dict:
        """Get learned relation importance weights (for interpretability)."""
        rel_names = ['nsubj', 'amod', 'obj', 'advmod', 'neg', 'compound', 'conj']
        weights = torch.sigmoid(self.rgat.relation_importance).detach().cpu().tolist()
        return {name: round(w, 4) for name, w in zip(rel_names, weights)}