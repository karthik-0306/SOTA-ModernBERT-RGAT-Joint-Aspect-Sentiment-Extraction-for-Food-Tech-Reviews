"""
ModernBERT-RGAT | Dataset & Preprocessing
===========================================
Handles:
  1. BIO label generation from aspect spans
  2. Sub-word alignment (first sub-token gets label, rest get -100)
  3. Dependency-based adjacency tensor creation (7 relations)
  4. Aspect mask generation for the ASC head
  5. PyTorch Dataset wrapping
"""

import re
import os
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
import pandas as pd
import spacy
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.data_pipeline import load_config


# ------------------------------------------------------------------
#  Preprocessor: Tokenization + Adjacency + BIO Alignment
# ------------------------------------------------------------------

class ABSAPreprocessor:
    """
    Preprocessing engine for the ModernBERT-RGAT pipeline.

    Handles ModernBERT tokenization with offset tracking,
    spaCy dependency parsing for 7-relation adjacency tensors,
    BIO label alignment to sub-word tokens, and aspect mask generation.
    """

    # 7 dependency relations relevant for ABSA sentiment flow
    RELATION_MAP = {
        'nsubj': 0,     # Nominal subject
        'amod': 1,      # Adjective modifier (key for sentiment)
        'obj': 2,       # Direct object
        'dobj': 2,      # Alias for obj
        'advmod': 3,    # Adverb modifier (key for sentiment)
        'neg': 4,       # Negation (critical)
        'compound': 5,  # Multi-word terms (ice cream)
        'conj': 6,      # Conjunctions (and, but)
    }
    NUM_RELATIONS = 7

    def __init__(self, model_name: str = "answerdotai/ModernBERT-base", max_len: int = 96):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")

        print(f"  Preprocessor initialized: {model_name}, max_len={max_len}")

    # -- Adjacency Tensor --------------------------------------------------

    def build_adjacency_tensor(self, text: str) -> torch.Tensor:
        """
        Build a [NUM_RELATIONS, max_len, max_len] adjacency tensor
        from spaCy dependency parse, aligned to ModernBERT sub-tokens.

        Returns:
            torch.Tensor of shape [7, max_len, max_len]
        """
        doc = self.nlp(text)

        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
        )
        offsets = encoding.offset_mapping

        # Map spaCy word index to list of BERT sub-token indices
        spacy_to_bert = [[] for _ in range(len(doc))]
        for b_idx, (start, end) in enumerate(offsets):
            if start == end:
                continue  # skip special tokens
            for word in doc:
                if word.idx <= start < (word.idx + len(word.text)):
                    spacy_to_bert[word.i].append(b_idx)
                    break

        # Build adjacency matrix
        adj = torch.zeros(self.NUM_RELATIONS, self.max_len, self.max_len)

        for token in doc:
            dep = token.dep_
            if dep in self.RELATION_MAP:
                rel_idx = self.RELATION_MAP[dep]
                child_indices = spacy_to_bert[token.i]
                head_indices = spacy_to_bert[token.head.i]

                for c_i in child_indices:
                    for h_i in head_indices:
                        if c_i < self.max_len and h_i < self.max_len:
                            adj[rel_idx, c_i, h_i] = 1
                            adj[rel_idx, h_i, c_i] = 1  # bidirectional

        return adj

    # -- BIO Label Alignment -----------------------------------------------

    def align_bio_labels(
        self,
        text: str,
        aspect: str,
        span_start: int,
        span_end: int,
        implicit_token: str = "[ASPECT]",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate BIO labels aligned to ModernBERT sub-tokens.

        Sub-word alignment strategy:
          - First sub-token of each word -> gets the real BIO label
          - Remaining sub-tokens of the same word -> -100 (ignored by loss)
          - Special tokens ([CLS], [SEP], [PAD]) -> -100

        Uses word_ids() from the tokenizer to detect word boundaries,
        which works correctly with GPT-style tokenizers (like ModernBERT)
        where leading spaces are part of the token.

        Also generates an aspect_mask for the ASC head (all aspect
        sub-tokens are marked, not just the first).

        Args:
            text:       Original sentence text
            aspect:     Aspect term string (or "[ASPECT]" for implicit)
            span_start: Character-level start index of aspect in text
            span_end:   Character-level end index of aspect in text
            implicit_token: Marker for implicit aspects

        Returns:
            bio_labels:  [max_len] tensor (0=O, 1=B-ASP, 2=I-ASP, -100=ignore)
            aspect_mask: [max_len] tensor (1 at aspect token positions, 0 elsewhere)
        """
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
        )
        offsets = encoding.offset_mapping
        word_ids = encoding.word_ids()  # None for special/pad tokens, int for real tokens

        bio_labels = torch.full((self.max_len,), -100, dtype=torch.long)
        aspect_mask = torch.zeros(self.max_len, dtype=torch.float)

        # Handle implicit aspects - no BIO tagging needed, no aspect mask
        if aspect == implicit_token or (span_start == 0 and span_end == 0 and aspect == implicit_token):
            for idx in range(self.max_len):
                if word_ids[idx] is not None:
                    bio_labels[idx] = 0  # O tag for all real tokens
            return bio_labels, aspect_mask

        # Determine which tokens are first sub-tokens of their word
        prev_word_id = None
        is_first_subtoken = []
        for idx in range(self.max_len):
            wid = word_ids[idx]
            if wid is None:
                is_first_subtoken.append(False)
            else:
                is_first_subtoken.append(wid != prev_word_id)
            prev_word_id = wid

        # Assign BIO labels
        aspect_started = False
        for idx in range(self.max_len):
            wid = word_ids[idx]
            if wid is None:
                continue  # special/pad token stays -100

            start, end = offsets[idx]

            # Check if this sub-token overlaps with the aspect span
            token_in_aspect = (start < span_end) and (end > span_start)

            if token_in_aspect:
                if is_first_subtoken[idx]:
                    if not aspect_started:
                        bio_labels[idx] = 1  # B-ASP
                        aspect_started = True
                    else:
                        bio_labels[idx] = 2  # I-ASP
                # Non-first sub-tokens of aspect words stay -100

                # Aspect mask: mark ALL aspect sub-tokens (for MaxPool in ASC head)
                aspect_mask[idx] = 1.0
            else:
                if is_first_subtoken[idx]:
                    bio_labels[idx] = 0  # O
                # Non-first sub-tokens of non-aspect words stay -100

        return bio_labels, aspect_mask

    # -- Full Encoding -----------------------------------------------------

    def encode(
        self,
        text: str,
        aspect: str,
        polarity: str,
        span_start: int,
        span_end: int,
        label_map: Dict[str, int],
        implicit_token: str = "[ASPECT]",
    ) -> Dict[str, torch.Tensor]:
        """
        Full encoding pipeline for a single sample.

        Returns dict with all tensors needed for the model:
            input_ids, attention_mask, adj_matrix,
            bio_labels, aspect_mask, sentiment_label
        """
        # 1. Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
        )

        # 2. Adjacency tensor
        adj_tensor = self.build_adjacency_tensor(text)

        # 3. BIO labels + aspect mask
        bio_labels, aspect_mask = self.align_bio_labels(
            text, aspect, span_start, span_end, implicit_token
        )

        # 4. Sentiment label
        sentiment_label = torch.tensor(label_map.get(polarity, 0), dtype=torch.long)

        return {
            'input_ids': encoding['input_ids'].squeeze(0),         # [max_len]
            'attention_mask': encoding['attention_mask'].squeeze(0), # [max_len]
            'adj_matrix': adj_tensor,                                # [7, max_len, max_len]
            'bio_labels': bio_labels,                                # [max_len]
            'aspect_mask': aspect_mask,                              # [max_len]
            'sentiment_label': sentiment_label,                      # scalar
        }


# ------------------------------------------------------------------
#  PyTorch Dataset
# ------------------------------------------------------------------

class ABSADataset(Dataset):
    """
    PyTorch Dataset for Joint ABSA.

    Each sample contains:
        - input_ids:       ModernBERT token IDs          [max_len]
        - attention_mask:  Attention mask                 [max_len]
        - adj_matrix:      Dependency adjacency tensor    [7, max_len, max_len]
        - bio_labels:      BIO labels (-100 for ignored)  [max_len]
        - aspect_mask:     1 at aspect positions          [max_len]
        - sentiment_label: Polarity class index           scalar
    """

    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: ABSAPreprocessor,
        label_map: Dict[str, int],
        implicit_token: str = "[ASPECT]",
    ):
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.label_map = label_map
        self.implicit_token = implicit_token

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        return self.preprocessor.encode(
            text=row['sentence'],
            aspect=row['aspect'],
            polarity=row['polarity'],
            span_start=int(row['span_start']),
            span_end=int(row['span_end']),
            label_map=self.label_map,
            implicit_token=self.implicit_token,
        )


# ------------------------------------------------------------------
#  DataLoader Factory
# ------------------------------------------------------------------

def create_dataloader(
    df: pd.DataFrame,
    preprocessor: ABSAPreprocessor,
    label_map: Dict[str, int],
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    implicit_token: str = "[ASPECT]",
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a DataFrame split.

    Args:
        df:            DataFrame with columns: sentence, aspect, polarity, span_start, span_end
        preprocessor:  ABSAPreprocessor instance
        label_map:     Dict mapping polarity strings to indices
        batch_size:    Batch size
        shuffle:       Whether to shuffle (True for train, False for val/test)
        num_workers:   DataLoader workers
        implicit_token: Implicit aspect marker

    Returns:
        DataLoader instance
    """
    dataset = ABSADataset(
        df=df,
        preprocessor=preprocessor,
        label_map=label_map,
        implicit_token=implicit_token,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )