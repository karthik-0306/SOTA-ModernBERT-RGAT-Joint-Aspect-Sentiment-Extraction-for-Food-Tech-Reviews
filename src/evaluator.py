"""
ModernBERT-RGAT | Evaluator
============================
Comprehensive evaluation engine for the joint ATE + ASC model.

Provides:
  - ATE Evaluation: strict F1 (exact span match), partial F1 (overlap),
    token-level P/R/F1
  - ASC Evaluation: accuracy, per-class F1, macro/weighted F1,
    confusion matrix
  - Joint Evaluation: combined ATE+ASC score
  - Error Analysis: misclassified examples with context
  - SOTA Comparison: published benchmarks from literature
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, field

from torch.utils.data import DataLoader
from torch.amp import autocast
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.model import ModernBERT_RGAT
from src.trainer import extract_spans_from_bio, compute_strict_f1


# ---------------------------------------------------------------------------
#  Data Classes for Results
# ---------------------------------------------------------------------------

@dataclass
class ATEMetrics:
    """Aspect Term Extraction metrics."""
    strict_precision: float = 0.0
    strict_recall: float = 0.0
    strict_f1: float = 0.0
    partial_precision: float = 0.0
    partial_recall: float = 0.0
    partial_f1: float = 0.0
    token_precision: float = 0.0
    token_recall: float = 0.0
    token_f1: float = 0.0
    num_pred_spans: int = 0
    num_gold_spans: int = 0

    def to_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}


@dataclass
class ASCMetrics:
    """Aspect Sentiment Classification metrics."""
    accuracy: float = 0.0
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    per_class_f1: Dict[str, float] = field(default_factory=dict)
    per_class_precision: Dict[str, float] = field(default_factory=dict)
    per_class_recall: Dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> dict:
        d = {k: round(v, 4) if isinstance(v, float) else v
             for k, v in self.__dict__.items()
             if k != 'confusion_matrix'}
        if self.confusion_matrix is not None:
            d['confusion_matrix'] = self.confusion_matrix.tolist()
        return d


@dataclass
class EvaluationResult:
    """Complete evaluation result for one dataset."""
    dataset_year: str
    ate: ATEMetrics = field(default_factory=ATEMetrics)
    asc: ASCMetrics = field(default_factory=ASCMetrics)
    combined_f1: float = 0.0
    num_samples: int = 0
    errors: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'dataset_year': self.dataset_year,
            'ate': self.ate.to_dict(),
            'asc': self.asc.to_dict(),
            'combined_f1': round(self.combined_f1, 4),
            'num_samples': self.num_samples,
            'num_errors': len(self.errors),
        }


# ---------------------------------------------------------------------------
#  Partial Match F1 (IoU-based)
# ---------------------------------------------------------------------------

def compute_partial_f1(
    pred_bio_batch: List[List[int]],
    gold_bio_batch: List[List[int]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute partial-match F1 for ATE.

    A predicted span is a partial match if its IoU (intersection over union)
    with any gold span exceeds the threshold.

    Args:
        pred_bio_batch: list of predicted BIO sequences
        gold_bio_batch: list of gold BIO sequences
        iou_threshold: minimum IoU for a match (default 0.5)

    Returns:
        dict with precision, recall, f1
    """
    total_pred = 0
    total_gold = 0
    total_matches = 0

    for pred_seq, gold_seq in zip(pred_bio_batch, gold_bio_batch):
        pred_spans = extract_spans_from_bio(pred_seq)
        gold_spans = extract_spans_from_bio(gold_seq)

        total_pred += len(pred_spans)
        total_gold += len(gold_spans)

        matched_gold = set()
        for ps, pe in pred_spans:
            pred_set = set(range(ps, pe))
            for gi, (gs, ge) in enumerate(gold_spans):
                if gi in matched_gold:
                    continue
                gold_set = set(range(gs, ge))
                intersection = len(pred_set & gold_set)
                union = len(pred_set | gold_set)
                if union > 0 and (intersection / union) >= iou_threshold:
                    total_matches += 1
                    matched_gold.add(gi)
                    break

    precision = total_matches / max(total_pred, 1)
    recall = total_matches / max(total_gold, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {'precision': precision, 'recall': recall, 'f1': f1}


def compute_token_level_metrics(
    pred_bio_batch: List[List[int]],
    gold_bio_batch: List[List[int]],
) -> Dict[str, float]:
    """
    Compute token-level P/R/F1 for aspect detection (binary: aspect vs non-aspect).
    """
    all_preds = []
    all_golds = []

    for pred_seq, gold_seq in zip(pred_bio_batch, gold_bio_batch):
        for p, g in zip(pred_seq, gold_seq):
            if g == -100:
                continue
            all_preds.append(1 if p > 0 else 0)  # B-ASP or I-ASP
            all_golds.append(1 if g > 0 else 0)

    if not all_preds:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    return {
        'precision': precision_score(all_golds, all_preds, zero_division=0),
        'recall': recall_score(all_golds, all_preds, zero_division=0),
        'f1': f1_score(all_golds, all_preds, zero_division=0),
    }


# ---------------------------------------------------------------------------
#  SOTA Benchmark Tables
# ---------------------------------------------------------------------------

SOTA_BENCHMARKS = {
    '2014': {
        'BERT-SPC':      {'ate_f1': None, 'asc_acc': 84.46, 'asc_f1': 76.98},
        'LCF-BERT':      {'ate_f1': None, 'asc_acc': 87.14, 'asc_f1': 80.31},
        'DualGCN-BERT':  {'ate_f1': 83.02, 'asc_acc': 84.27, 'asc_f1': 78.08},
        'GRACE':         {'ate_f1': 87.93, 'asc_acc': None,  'asc_f1': 72.30},
        'InstructABSA':  {'ate_f1': 86.63, 'asc_acc': 90.18, 'asc_f1': 83.50},
        'SPAN-ASTE':     {'ate_f1': 86.71, 'asc_acc': None,  'asc_f1': None},
        'MVP':           {'ate_f1': 85.30, 'asc_acc': None,  'asc_f1': 75.41},
    },
    '2015': {
        'BERT-SPC':      {'ate_f1': None, 'asc_acc': 81.18, 'asc_f1': 60.32},
        'LCF-BERT':      {'ate_f1': None, 'asc_acc': 82.45, 'asc_f1': 63.89},
        'DualGCN-BERT':  {'ate_f1': 67.89, 'asc_acc': 81.16, 'asc_f1': 62.32},
        'GRACE':         {'ate_f1': 72.75, 'asc_acc': None,  'asc_f1': 57.63},
        'InstructABSA':  {'ate_f1': 74.83, 'asc_acc': 85.22, 'asc_f1': 68.10},
    },
    '2016': {
        'BERT-SPC':      {'ate_f1': None, 'asc_acc': 87.89, 'asc_f1': 66.75},
        'LCF-BERT':      {'ate_f1': None, 'asc_acc': 89.72, 'asc_f1': 73.18},
        'DualGCN-BERT':  {'ate_f1': 74.81, 'asc_acc': 88.36, 'asc_f1': 70.92},
        'GRACE':         {'ate_f1': 78.93, 'asc_acc': None,  'asc_f1': 66.38},
        'InstructABSA':  {'ate_f1': 79.45, 'asc_acc': 91.04, 'asc_f1': 76.38},
    },
}


def get_sota_comparison_df(year: str, our_metrics: dict) -> pd.DataFrame:
    """
    Build a SOTA comparison table for the given year.

    Args:
        year: '2014', '2015', or '2016'
        our_metrics: dict with ate_f1, asc_acc, asc_f1

    Returns:
        DataFrame with one row per model including ours
    """
    benchmarks = SOTA_BENCHMARKS.get(year, {})
    rows = []
    for model_name, scores in benchmarks.items():
        rows.append({
            'Model': model_name,
            'ATE F1': f"{scores['ate_f1']:.2f}" if scores['ate_f1'] else '—',
            'ASC Acc': f"{scores['asc_acc']:.2f}" if scores['asc_acc'] else '—',
            'ASC F1': f"{scores['asc_f1']:.2f}" if scores['asc_f1'] else '—',
        })

    # Add our model
    rows.append({
        'Model': '⭐ Ours (ModernBERT-RGAT)',
        'ATE F1': f"{our_metrics.get('ate_f1', 0)*100:.2f}",
        'ASC Acc': f"{our_metrics.get('asc_acc', 0)*100:.2f}",
        'ASC F1': f"{our_metrics.get('asc_f1', 0)*100:.2f}",
    })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
#  Model Evaluator
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """
    Comprehensive evaluator for the ModernBERT-RGAT model.

    Loads a checkpoint, runs inference on a DataLoader, and computes
    all ATE, ASC, and joint metrics.

    Args:
        model: ModernBERT_RGAT instance (architecture only, weights loaded from checkpoint)
        device: torch device
        label_names: list of sentiment label names in order (e.g., ['positive', 'negative', 'neutral', 'conflict'])
    """

    def __init__(
        self,
        model: ModernBERT_RGAT,
        device: torch.device,
        label_names: List[str],
    ):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.label_names = label_names
        self.use_fp16 = device.type == 'cuda'

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: dict,
        device: torch.device,
    ) -> 'ModelEvaluator':
        """
        Create evaluator from a saved checkpoint.

        Args:
            checkpoint_path: path to .pt checkpoint file
            config: full config dict
            device: torch device

        Returns:
            ModelEvaluator instance with loaded weights
        """
        # Build model architecture
        model = ModernBERT_RGAT(
            model_name=config['model']['backbone'],
            hidden_dim=config['model']['hidden_dim'],
            num_sentiment_classes=config['model']['num_sentiment_classes'],
            num_bio_tags=config['model']['num_bio_tags'],
            num_relations=config['model']['rgat']['num_relations'],
            rgat_dropout=config['model']['rgat']['dropout'],
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"  Loaded checkpoint: {checkpoint_path}")
        print(f"  Trained for {checkpoint.get('epoch', '?')} epochs")

        val_metrics = checkpoint.get('val_metrics', {})
        if val_metrics:
            print(f"  Val ATE F1: {val_metrics.get('ate_f1', 0):.4f}")
            print(f"  Val ASC F1: {val_metrics.get('asc_f1', 0):.4f}")

        # Build label names
        label_map = config['labels']['polarity']
        label_names = [name for name, _ in sorted(label_map.items(), key=lambda x: x[1])]

        return cls(model=model, device=device, label_names=label_names)

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        dataset_year: str = "2014",
        collect_errors: bool = True,
        max_errors: int = 50,
    ) -> EvaluationResult:
        """
        Run comprehensive evaluation on a dataset.

        Args:
            dataloader: test DataLoader
            dataset_year: year identifier for reporting
            collect_errors: whether to collect misclassified examples
            max_errors: max number of error examples to keep

        Returns:
            EvaluationResult with all metrics and error analysis
        """
        self.model.eval()

        all_ate_preds = []
        all_ate_labels = []
        all_asc_preds = []
        all_asc_labels = []
        all_input_ids = []
        all_attention_masks = []

        for batch in dataloader:
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

            ate_preds = outputs["ate_logits"].argmax(dim=-1)
            asc_preds = outputs["sentiment_logits"].argmax(dim=-1)

            # Collect predictions per sample
            for i in range(ate_preds.size(0)):
                pred_seq = ate_preds[i].cpu().tolist()
                gold_seq = bio_labels[i].cpu().tolist()

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
            all_input_ids.extend(input_ids.cpu().tolist())
            all_attention_masks.extend(attention_mask.cpu().tolist())

        # --- Compute ATE Metrics ---
        strict = compute_strict_f1(all_ate_preds, all_ate_labels)
        partial = compute_partial_f1(all_ate_preds, all_ate_labels, iou_threshold=0.5)
        token = compute_token_level_metrics(all_ate_preds, all_ate_labels)

        # Count total predicted and gold spans
        num_pred = sum(len(extract_spans_from_bio(s)) for s in all_ate_preds)
        num_gold = sum(len(extract_spans_from_bio(s)) for s in all_ate_labels)

        ate_metrics = ATEMetrics(
            strict_precision=strict['precision'],
            strict_recall=strict['recall'],
            strict_f1=strict['f1'],
            partial_precision=partial['precision'],
            partial_recall=partial['recall'],
            partial_f1=partial['f1'],
            token_precision=token['precision'],
            token_recall=token['recall'],
            token_f1=token['f1'],
            num_pred_spans=num_pred,
            num_gold_spans=num_gold,
        )

        # --- Compute ASC Metrics ---
        asc_preds_arr = np.array(all_asc_preds)
        asc_labels_arr = np.array(all_asc_labels)

        asc_accuracy = accuracy_score(asc_labels_arr, asc_preds_arr)
        asc_macro_f1 = f1_score(asc_labels_arr, asc_preds_arr, average='macro', zero_division=0)
        asc_weighted_f1 = f1_score(asc_labels_arr, asc_preds_arr, average='weighted', zero_division=0)

        # Per-class metrics
        per_class_f1 = {}
        per_class_p = {}
        per_class_r = {}
        unique_labels = sorted(set(asc_labels_arr.tolist()) | set(asc_preds_arr.tolist()))
        for label_idx in unique_labels:
            if label_idx < len(self.label_names):
                name = self.label_names[label_idx]
            else:
                name = f"class_{label_idx}"
            binary_gold = (asc_labels_arr == label_idx).astype(int)
            binary_pred = (asc_preds_arr == label_idx).astype(int)
            per_class_f1[name] = f1_score(binary_gold, binary_pred, zero_division=0)
            per_class_p[name] = precision_score(binary_gold, binary_pred, zero_division=0)
            per_class_r[name] = recall_score(binary_gold, binary_pred, zero_division=0)

        cm = confusion_matrix(
            asc_labels_arr, asc_preds_arr,
            labels=list(range(len(self.label_names))),
        )

        asc_metrics = ASCMetrics(
            accuracy=asc_accuracy,
            macro_f1=asc_macro_f1,
            weighted_f1=asc_weighted_f1,
            per_class_f1=per_class_f1,
            per_class_precision=per_class_p,
            per_class_recall=per_class_r,
            confusion_matrix=cm,
        )

        # --- Combined Score ---
        combined_f1 = (ate_metrics.strict_f1 + asc_metrics.macro_f1) / 2

        # --- Error Analysis ---
        errors = []
        if collect_errors:
            errors = self._collect_errors(
                all_ate_preds, all_ate_labels,
                all_asc_preds, all_asc_labels,
                all_input_ids, all_attention_masks,
                max_errors=max_errors,
            )

        result = EvaluationResult(
            dataset_year=dataset_year,
            ate=ate_metrics,
            asc=asc_metrics,
            combined_f1=combined_f1,
            num_samples=len(all_asc_preds),
            errors=errors,
        )

        return result

    def _collect_errors(
        self,
        ate_preds_batch, ate_labels_batch,
        asc_preds, asc_labels,
        input_ids_batch, attention_masks_batch,
        max_errors: int = 50,
    ) -> List[dict]:
        """Collect misclassified examples for error analysis."""
        errors = []

        for i in range(len(asc_preds)):
            ate_pred = ate_preds_batch[i]
            ate_gold = ate_labels_batch[i]
            asc_pred = asc_preds[i]
            asc_gold = asc_labels[i]

            # Check for ATE errors (span mismatch)
            pred_spans = extract_spans_from_bio(ate_pred)
            gold_spans = extract_spans_from_bio(ate_gold)
            ate_error = set(pred_spans) != set(gold_spans)

            # Check for ASC errors
            asc_error = asc_pred != asc_gold

            if ate_error or asc_error:
                error_entry = {
                    'sample_idx': i,
                    'ate_error': ate_error,
                    'asc_error': asc_error,
                    'pred_spans': pred_spans,
                    'gold_spans': gold_spans,
                    'pred_sentiment': self.label_names[asc_pred] if asc_pred < len(self.label_names) else f'class_{asc_pred}',
                    'gold_sentiment': self.label_names[asc_gold] if asc_gold < len(self.label_names) else f'class_{asc_gold}',
                    'error_type': self._classify_error(
                        ate_error, asc_error, pred_spans, gold_spans
                    ),
                }
                errors.append(error_entry)

                if len(errors) >= max_errors:
                    break

        return errors

    @staticmethod
    def _classify_error(
        ate_error: bool,
        asc_error: bool,
        pred_spans: list,
        gold_spans: list,
    ) -> str:
        """Classify the type of error for analysis."""
        if ate_error and asc_error:
            return 'both_wrong'
        elif asc_error:
            return 'sentiment_only'
        elif ate_error:
            # Sub-classify ATE errors
            if not pred_spans and gold_spans:
                return 'missed_aspect'
            elif pred_spans and not gold_spans:
                return 'false_aspect'
            elif len(pred_spans) != len(gold_spans):
                return 'wrong_count'
            else:
                return 'wrong_boundary'
        return 'unknown'


def get_error_summary(errors: List[dict]) -> pd.DataFrame:
    """
    Summarize error types into a distribution table.

    Args:
        errors: list of error dicts from ModelEvaluator

    Returns:
        DataFrame with error type counts and percentages
    """
    if not errors:
        return pd.DataFrame(columns=['Error Type', 'Count', 'Percentage'])

    counter = Counter(e['error_type'] for e in errors)
    total = sum(counter.values())

    rows = []
    for error_type, count in counter.most_common():
        rows.append({
            'Error Type': error_type,
            'Count': count,
            'Percentage': f"{100 * count / total:.1f}%",
        })
    return pd.DataFrame(rows)


def get_sentiment_confusion_analysis(errors: List[dict]) -> pd.DataFrame:
    """
    Analyze sentiment confusion patterns from errors.

    Returns DataFrame showing which sentiment pairs are most often confused.
    """
    asc_errors = [e for e in errors if e['asc_error']]
    if not asc_errors:
        return pd.DataFrame(columns=['Gold', 'Predicted', 'Count'])

    confusion_pairs = Counter(
        (e['gold_sentiment'], e['pred_sentiment']) for e in asc_errors
    )

    rows = []
    for (gold, pred), count in confusion_pairs.most_common():
        rows.append({'Gold': gold, 'Predicted': pred, 'Count': count})
    return pd.DataFrame(rows)


def save_evaluation_results(
    results: Dict[str, EvaluationResult],
    output_dir: str = "outputs/results",
):
    """
    Save evaluation results for all datasets to JSON files.

    Args:
        results: dict mapping year -> EvaluationResult
        output_dir: directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)

    for year, result in results.items():
        filepath = os.path.join(output_dir, f"metrics_{year}.json")
        with open(filepath, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"  Saved: {filepath}")

    # Combined summary
    summary = {year: result.to_dict() for year, result in results.items()}
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")
