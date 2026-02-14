"""
ModernBERT-RGAT | Data Pipeline
================================
Handles data loading, stratified splitting, caching, and DataLoader creation.
All operations work with Processed_Data CSVs only.
"""

import os
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


# ══════════════════════════════════════════════════════════════════
#  Config Loader
# ══════════════════════════════════════════════════════════════════

def load_config(config_path: str = "configs/config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ══════════════════════════════════════════════════════════════════
#  Data Loader
# ══════════════════════════════════════════════════════════════════

def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load a processed CSV dataset with standardised column ordering.
    
    Handles the column ordering difference between 2014 and 2015/2016:
      - 2014: sentence_id, sentence, aspect, polarity, from, to
      - 2015/2016: sentence_id, sentence, aspect, from, to, polarity
    
    Returns DataFrame with columns:
      sentence_id, sentence, aspect, polarity, span_start, span_end
    """
    df = pd.read_csv(csv_path)

    # Standardise column names
    rename_map = {"from": "span_start", "to": "span_end"}
    df = df.rename(columns=rename_map)

    # Ensure consistent column order
    standard_cols = ["sentence_id", "sentence", "aspect", "polarity", "span_start", "span_end"]
    df = df[standard_cols]

    return df


def load_all_datasets(config: dict) -> Dict[str, pd.DataFrame]:
    """Load all three processed datasets and return as a dict keyed by year."""
    processed_dir = config["data"]["processed_dir"]
    datasets = {}

    for filename in config["data"]["datasets"]:
        year = filename.split("_")[0]          # "2014", "2015", "2016"
        path = os.path.join(processed_dir, filename)
        datasets[year] = load_dataset(path)
        print(f"  Loaded {year}: {len(datasets[year]):,} rows, "
              f"{datasets[year]['sentence_id'].nunique():,} sentences")

    return datasets


# ══════════════════════════════════════════════════════════════════
#  Stratified Sentence-Level Splitting
# ══════════════════════════════════════════════════════════════════

def _get_majority_polarity(df: pd.DataFrame) -> pd.Series:
    """
    For each sentence, get its 'majority polarity' to use as the
    stratification target.  When a sentence has multiple aspects with
    different polarities, we pick the most common one.
    """
    return (
        df.groupby("sentence_id")["polarity"]
        .agg(lambda x: x.value_counts().index[0])
    )


def stratified_sentence_split(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data at the **sentence level** with stratification by polarity.
    
    1.  Extract unique sentence IDs with their majority polarity.
    2.  Stratified split of sentence IDs → train / val / test.
    3.  Filter original rows by those IDs.
    
    This prevents data leakage: no sentence appears in more than one split.
    
    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    # Step 1: majority polarity per sentence
    sent_polarity = _get_majority_polarity(df).reset_index()
    sent_polarity.columns = ["sentence_id", "majority_polarity"]

    sentence_ids = sent_polarity["sentence_id"].values
    labels = sent_polarity["majority_polarity"].values

    # Step 2: first split → train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_ids, valtest_ids, train_labels, valtest_labels = train_test_split(
        sentence_ids, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Step 3: second split → val vs test
    relative_test = test_ratio / val_test_ratio
    val_ids, test_ids, _, _ = train_test_split(
        valtest_ids, valtest_labels,
        test_size=relative_test,
        stratify=valtest_labels,
        random_state=seed,
    )

    # Step 4: filter rows
    train_df = df[df["sentence_id"].isin(train_ids)].reset_index(drop=True)
    val_df   = df[df["sentence_id"].isin(val_ids)].reset_index(drop=True)
    test_df  = df[df["sentence_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, val_df, test_df


# ══════════════════════════════════════════════════════════════════
#  Split Validation / Diagnostics
# ══════════════════════════════════════════════════════════════════

def validate_split(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str = "",
) -> Dict:
    """
    Validate the data split for correctness:
      - No sentence overlap between splits
      - Sentiment distribution is preserved (stratification check)
    
    Returns a summary dict.
    """
    train_sids = set(train_df["sentence_id"].unique())
    val_sids   = set(val_df["sentence_id"].unique())
    test_sids  = set(test_df["sentence_id"].unique())

    # Overlap checks
    overlap_tv = train_sids & val_sids
    overlap_tt = train_sids & test_sids
    overlap_vt = val_sids & test_sids

    leakage = len(overlap_tv) + len(overlap_tt) + len(overlap_vt)

    # Distribution
    total = len(train_df) + len(val_df) + len(test_df)

    def _dist(split_df):
        return split_df["polarity"].value_counts(normalize=True).to_dict()

    summary = {
        "dataset": dataset_name,
        "total_rows": total,
        "train": {"rows": len(train_df), "sentences": len(train_sids)},
        "val":   {"rows": len(val_df),   "sentences": len(val_sids)},
        "test":  {"rows": len(test_df),  "sentences": len(test_sids)},
        "leakage_count": leakage,
        "train_dist": _dist(train_df),
        "val_dist":   _dist(val_df),
        "test_dist":  _dist(test_df),
    }

    return summary


def print_split_summary(summary: Dict) -> None:
    """Pretty-print a split validation summary."""
    name = summary["dataset"]
    print(f"\n{'='*70}")
    print(f"  SPLIT SUMMARY: {name}")
    print(f"{'='*70}")
    print(f"  Total rows: {summary['total_rows']:,}")
    print()

    for split in ["train", "val", "test"]:
        s = summary[split]
        print(f"  {split.upper():5s}:  {s['rows']:5d} rows  |  {s['sentences']:4d} sentences")

    print()
    leak = summary["leakage_count"]
    status = " NO LEAKAGE" if leak == 0 else f" LEAKAGE DETECTED ({leak} overlapping sentences)"
    print(f"  Data Leakage Check: {status}")

    print(f"\n  Polarity Distribution (%):")
    print(f"  {'':12s} {'Train':>8s} {'Val':>8s} {'Test':>8s}")
    all_labels = sorted(
        set(list(summary["train_dist"].keys()) +
            list(summary["val_dist"].keys()) +
            list(summary["test_dist"].keys()))
    )
    for label in all_labels:
        tr = summary["train_dist"].get(label, 0) * 100
        va = summary["val_dist"].get(label, 0) * 100
        te = summary["test_dist"].get(label, 0) * 100
        print(f"  {label:12s} {tr:7.1f}% {va:7.1f}% {te:7.1f}%")

    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════
#  Caching Utilities
# ══════════════════════════════════════════════════════════════════

def _cache_key(csv_path: str, config: dict) -> str:
    """Create a deterministic cache key from the file path + split config."""
    seed = config["data"]["split"]["seed"]
    ratios = (
        config["data"]["split"]["train"],
        config["data"]["split"]["val"],
        config["data"]["split"]["test"],
    )
    raw = f"{csv_path}|{seed}|{ratios}"
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def save_splits_to_cache(
    splits: Dict[str, pd.DataFrame],
    cache_dir: str,
    key: str,
) -> None:
    """Pickle splits to cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"splits_{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(splits, f)
    print(f"  Cached splits -> {path}")


def load_splits_from_cache(cache_dir: str, key: str) -> Optional[Dict[str, pd.DataFrame]]:
    """Load cached splits if they exist."""
    path = os.path.join(cache_dir, f"splits_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            splits = pickle.load(f)
        print(f"  Loaded cached splits from {path}")
        return splits
    return None


# ══════════════════════════════════════════════════════════════════
#  Class Weights (for imbalanced sentiments)
# ══════════════════════════════════════════════════════════════════

def compute_class_weights(df: pd.DataFrame, label_map: dict) -> Dict[int, float]:
    """
    Compute inverse-frequency class weights for the polarity labels.
    
    Returns dict mapping label_index → weight.
    """
    counts = df["polarity"].value_counts()
    total = counts.sum()
    n_classes = len(label_map)

    weights = {}
    for label_name, label_idx in label_map.items():
        count = counts.get(label_name, 1)  # avoid div-by-zero
        weights[label_idx] = total / (n_classes * count)

    return weights


# ══════════════════════════════════════════════════════════════════
#  Main Pipeline Entry Point
# ══════════════════════════════════════════════════════════════════

def build_splits(
    config: dict,
    year: str,
    use_cache: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    End-to-end pipeline for a single dataset year:
    
    1. Load the processed CSV
    2. Attempt to load from cache
    3. If not cached, perform stratified sentence-level split
    4. Validate the split
    5. Cache the result
    
    Args:
        config:    loaded YAML config dict
        year:      "2014", "2015", or "2016"
        use_cache: whether to use/save cache
        verbose:   print diagnostics
    
    Returns:
        (train_df, val_df, test_df)
    """
    # Resolve file path
    processed_dir = config["data"]["processed_dir"]
    filename = f"{year}_rest_reviews.csv"
    csv_path = os.path.join(processed_dir, filename)

    cache_dir = config["data"].get("cache_dir", "Data/cached")
    key = _cache_key(csv_path, config)

    # Try cache first
    if use_cache:
        cached = load_splits_from_cache(cache_dir, key)
        if cached is not None:
            return cached["train"], cached["val"], cached["test"]

    # Load data
    if verbose:
        print(f"\n Loading {csv_path} ...")
    df = load_dataset(csv_path)
    if verbose:
        print(f"   {len(df):,} rows, {df['sentence_id'].nunique():,} unique sentences")

    # Split
    split_cfg = config["data"]["split"]
    train_df, val_df, test_df = stratified_sentence_split(
        df,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
        seed=split_cfg["seed"],
    )

    # Validate
    summary = validate_split(train_df, val_df, test_df, dataset_name=f"SemEval {year}")
    if verbose:
        print_split_summary(summary)

    # Cache
    if use_cache:
        save_splits_to_cache(
            {"train": train_df, "val": val_df, "test": test_df},
            cache_dir, key,
        )

    return train_df, val_df, test_df


def build_all_splits(config: dict, verbose: bool = True) -> Dict[str, Tuple]:
    """
    Build train/val/test splits for all datasets defined in the config.
    
    Returns:
        {"2014": (train, val, test), "2015": (...), "2016": (...)}
    """
    all_splits = {}
    for filename in config["data"]["datasets"]:
        year = filename.split("_")[0]
        train_df, val_df, test_df = build_splits(config, year, verbose=verbose)
        all_splits[year] = (train_df, val_df, test_df)
    return all_splits


# -- Quick Sanity Test --------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    print(" Building splits for all datasets ...\n")
    all_splits = build_all_splits(cfg)

    for year, (tr, va, te) in all_splits.items():
        print(f"  {year} — train: {len(tr)}, val: {len(va)}, test: {len(te)}")

    # Class weights example
    label_map = cfg["labels"]["polarity"]
    for year, (tr, _, _) in all_splits.items():
        weights = compute_class_weights(tr, label_map)
        print(f"\n  {year} class weights: {weights}")
