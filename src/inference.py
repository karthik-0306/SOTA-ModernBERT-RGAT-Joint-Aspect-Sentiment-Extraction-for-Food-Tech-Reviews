"""
ModernBERT-RGAT | Inference Pipeline
======================================
End-to-end inference for Joint Aspect Extraction & Sentiment Classification.

Usage:
    >>> from src.inference import AspectSentimentPredictor
    >>> predictor = AspectSentimentPredictor.from_checkpoint('checkpoints/best_model_2014.pt')
    >>> results = predictor.predict("The pasta was delicious but service was terrible")
    >>> print(results)
    [{'aspect': 'pasta', 'sentiment': 'positive', 'confidence': 0.92, 'start': 4, 'end': 9},
     {'aspect': 'service', 'sentiment': 'negative', 'confidence': 0.88, 'start': 34, 'end': 41}]
"""

import os
import torch
import spacy
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer

from src.model import ModernBERT_RGAT
from src.trainer import extract_spans_from_bio


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class AspectPrediction:
    """A single extracted aspect with its sentiment."""
    aspect: str
    sentiment: str
    confidence: float
    start: int  # char offset in original text
    end: int    # char offset in original text

    def to_dict(self) -> dict:
        return {
            'aspect': self.aspect,
            'sentiment': self.sentiment,
            'confidence': round(self.confidence, 4),
            'start': self.start,
            'end': self.end,
        }


# ---------------------------------------------------------------------------
#  Relation Map (same as in dataset.py)
# ---------------------------------------------------------------------------

RELATION_MAP = {
    'nsubj': 0, 'amod': 1, 'dobj': 2, 'obj': 2,
    'advmod': 3, 'neg': 4, 'compound': 5, 'conj': 6,
}
NUM_RELATIONS = 7


# ---------------------------------------------------------------------------
#  Aspect-Sentiment Predictor
# ---------------------------------------------------------------------------

class AspectSentimentPredictor:
    """
    End-to-end inference engine for ModernBERT-RGAT.

    Takes raw text, extracts aspect terms, and classifies their sentiment.

    Args:
        model: ModernBERT_RGAT instance with loaded weights
        tokenizer: HuggingFace tokenizer
        nlp: spaCy language model
        label_names: ordered list of sentiment labels
        device: torch device
        max_len: max sequence length
    """

    def __init__(
        self,
        model: ModernBERT_RGAT,
        tokenizer,
        nlp,
        label_names: List[str],
        device: torch.device,
        max_len: int = 96,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.label_names = label_names
        self.device = device
        self.max_len = max_len

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config_path: str = None,
        config: dict = None,
        device: torch.device = None,
    ) -> 'AspectSentimentPredictor':
        """
        Create predictor from a saved checkpoint.

        Args:
            checkpoint_path: path to .pt file
            config_path: path to config.yaml (optional if config dict provided)
            config: config dict (optional if config_path provided)
            device: torch device (auto-detected if None)

        Returns:
            AspectSentimentPredictor ready for prediction
        """
        # Device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Config
        if config is None:
            from src.data_pipeline import load_config
            if config_path is None:
                # Try default location
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                config_path = os.path.join(project_root, 'configs', 'config.yaml')
            config = load_config(config_path)

        # Build model
        model = ModernBERT_RGAT(
            model_name=config['model']['backbone'],
            hidden_dim=config['model']['hidden_dim'],
            num_sentiment_classes=config['model']['num_sentiment_classes'],
            num_bio_tags=config['model']['num_bio_tags'],
            num_relations=config['model']['rgat']['num_relations'],
            rgat_dropout=0.0,  # No dropout at inference
        )

        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', '?')}")

        # Tokenizer and spaCy
        tokenizer = AutoTokenizer.from_pretrained(config['model']['backbone'])
        nlp = spacy.load('en_core_web_sm')

        # Labels
        label_map = config['labels']['polarity']
        label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

        return cls(
            model=model,
            tokenizer=tokenizer,
            nlp=nlp,
            label_names=label_names,
            device=device,
            max_len=config['model']['max_len'],
        )

    def _build_adjacency_tensor(self, text: str) -> torch.Tensor:
        """Build spaCy dependency adjacency tensor aligned to BERT tokens."""
        doc = self.nlp(text)

        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )
        offsets = encoding.offset_mapping

        # Map spaCy word → BERT sub-token indices
        spacy_to_bert = [[] for _ in range(len(doc))]
        for b_idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            for word in doc:
                if word.idx <= start < (word.idx + len(word.text)):
                    spacy_to_bert[word.i].append(b_idx)
                    break

        # Build adjacency
        adj = torch.zeros(NUM_RELATIONS, self.max_len, self.max_len)
        for token in doc:
            dep = token.dep_
            if dep in RELATION_MAP:
                rel_idx = RELATION_MAP[dep]
                child_indices = spacy_to_bert[token.i]
                head_indices = spacy_to_bert[token.head.i]
                for c_i in child_indices:
                    for h_i in head_indices:
                        if c_i < self.max_len and h_i < self.max_len:
                            adj[rel_idx, c_i, h_i] = 1
                            adj[rel_idx, h_i, c_i] = 1

        return adj

    def _tokenize(self, text: str) -> dict:
        """Tokenize text and build all model inputs."""
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True,
        )

        offsets = encoding.pop('offset_mapping')[0].tolist()
        adj = self._build_adjacency_tensor(text).unsqueeze(0)

        return {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'adj_matrix': adj,
            'offsets': offsets,
        }

    def _decode_aspects(
        self,
        bio_preds: List[int],
        offsets: List[Tuple[int, int]],
        text: str,
    ) -> List[dict]:
        """
        Decode BIO predictions into aspect text spans.

        Returns list of dicts with aspect text, start/end char positions.
        """
        spans = extract_spans_from_bio(bio_preds)
        aspects = []

        for tok_start, tok_end in spans:
            # Map token positions to character positions
            char_start = None
            char_end = None

            for tok_idx in range(tok_start, tok_end):
                if tok_idx < len(offsets):
                    off_s, off_e = offsets[tok_idx]
                    if off_s == off_e:
                        continue
                    if char_start is None:
                        char_start = off_s
                    char_end = off_e

            if char_start is not None and char_end is not None:
                aspect_text = text[char_start:char_end].strip()
                if aspect_text:
                    aspects.append({
                        'text': aspect_text,
                        'start': char_start,
                        'end': char_end,
                        'token_span': (tok_start, tok_end),
                    })

        return aspects

    @torch.no_grad()
    def predict(self, text: str) -> List[AspectPrediction]:
        """
        Extract aspects and predict sentiments from raw text.

        Args:
            text: raw input text (e.g., restaurant review)

        Returns:
            List of AspectPrediction objects
        """
        # Tokenize
        inputs = self._tokenize(text)
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        adj_matrix = inputs['adj_matrix'].to(self.device)
        offsets = inputs['offsets']

        # Forward pass (no aspect mask = inference mode)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            adj_matrix=adj_matrix,
            aspect_mask=None,  # Model derives from ATE predictions
        )

        ate_logits = outputs['ate_logits'][0]  # [seq_len, 3]
        ate_probs = torch.softmax(ate_logits, dim=-1)
        ate_preds = ate_logits.argmax(dim=-1).cpu().tolist()

        sentiment_logits = outputs['sentiment_logits'][0]  # [num_classes]
        sentiment_probs = torch.softmax(sentiment_logits, dim=-1)

        # Decode aspects
        aspects = self._decode_aspects(ate_preds, offsets, text)

        if not aspects:
            return []

        # For each extracted aspect, get its sentiment
        # The model provides a single sentence-level sentiment based on
        # all detected aspects. For per-aspect sentiment, we re-run with
        # individual aspect masks.
        predictions = []

        for aspect_info in aspects:
            tok_start, tok_end = aspect_info['token_span']

            # Build aspect-specific mask
            aspect_mask = torch.zeros(1, self.max_len, device=self.device)
            for t in range(tok_start, tok_end):
                if t < self.max_len:
                    aspect_mask[0, t] = 1.0

            # Re-run model with specific aspect mask for per-aspect sentiment
            aspect_outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                adj_matrix=adj_matrix,
                aspect_mask=aspect_mask,
            )

            asp_logits = aspect_outputs['sentiment_logits'][0]
            asp_probs = torch.softmax(asp_logits, dim=-1)
            pred_idx = asp_probs.argmax().item()
            confidence = asp_probs[pred_idx].item()

            predictions.append(AspectPrediction(
                aspect=aspect_info['text'],
                sentiment=self.label_names[pred_idx],
                confidence=confidence,
                start=aspect_info['start'],
                end=aspect_info['end'],
            ))

        return predictions

    def predict_batch(self, texts: List[str]) -> List[List[AspectPrediction]]:
        """
        Predict aspects and sentiments for multiple texts.

        Args:
            texts: list of input texts

        Returns:
            List of prediction lists (one per input text)
        """
        return [self.predict(text) for text in texts]

    def format_predictions(self, text: str, predictions: List[AspectPrediction]) -> str:
        """
        Format predictions into a readable string with highlighted aspects.

        Args:
            text: original input text
            predictions: list of AspectPrediction objects

        Returns:
            Formatted string
        """
        if not predictions:
            return f'Text: "{text}"\n  No aspects detected.'

        lines = [f'Text: "{text}"']
        lines.append(f'  Found {len(predictions)} aspect(s):')

        # Sentiment emoji
        emoji = {
            'positive': '[+]', 'negative': '[-]',
            'neutral': '[~]', 'conflict': '[?]',
        }

        for p in predictions:
            e = emoji.get(p.sentiment, '[ ]')
            lines.append(
                f'    {e} "{p.aspect}" → {p.sentiment} '
                f'(conf: {p.confidence:.2f}, chars {p.start}:{p.end})'
            )

        return '\n'.join(lines)

    def get_highlighted_html(
        self,
        text: str,
        predictions: List[AspectPrediction],
    ) -> str:
        """
        Generate HTML with color-coded aspect highlights.

        Args:
            text: original text
            predictions: list of AspectPrediction objects

        Returns:
            HTML string with colored spans
        """
        colors = {
            'positive': '#27ae60',
            'negative': '#e74c3c',
            'neutral': '#f39c12',
            'conflict': '#8e44ad',
        }

        # Sort predictions by start position
        sorted_preds = sorted(predictions, key=lambda p: p.start)

        html_parts = []
        last_end = 0

        for pred in sorted_preds:
            # Text before this aspect
            if pred.start > last_end:
                html_parts.append(text[last_end:pred.start])

            # Highlighted aspect
            color = colors.get(pred.sentiment, '#95a5a6')
            html_parts.append(
                f'<span style="background-color: {color}; color: white; '
                f'padding: 2px 6px; border-radius: 4px; font-weight: bold;" '
                f'title="{pred.sentiment} ({pred.confidence:.2f})">'
                f'{pred.aspect}</span>'
            )
            last_end = pred.end

        # Remaining text
        if last_end < len(text):
            html_parts.append(text[last_end:])

        return ''.join(html_parts)


# ---------------------------------------------------------------------------
#  Convenience functions
# ---------------------------------------------------------------------------

def load_predictor(
    checkpoint_path: str = None,
    year: str = '2014',
    device: torch.device = None,
) -> AspectSentimentPredictor:
    """
    Quick-load a predictor from the default checkpoint directory.

    Args:
        checkpoint_path: explicit path, or auto-resolve from year
        year: dataset year to load (2014, 2015, 2016)
        device: torch device

    Returns:
        Ready-to-use AspectSentimentPredictor
    """
    if checkpoint_path is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        checkpoint_path = os.path.join(
            project_root, 'checkpoints', f'best_model_{year}.pt'
        )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Train the model first using notebooks/04_training.ipynb"
        )

    return AspectSentimentPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
    )
