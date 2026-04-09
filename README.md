---
title: ABSA Restaurant Reviews
emoji: 📊
colorFrom: purple
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Aspect extraction and sentiment for restaurant reviews
---

# ModernBERT-RGAT: Joint Aspect Extraction & Sentiment Classification

<div align="center">

**An end-to-end deep learning engine for aspect-level sentiment analysis in restaurant reviews**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org)
[![ModernBERT](https://img.shields.io/badge/Backbone-ModernBERT_base-orange.svg)](https://huggingface.co/answerdotai/ModernBERT-base)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## 📌 Overview

This project implements a novel **joint model** that simultaneously performs:

1. **Aspect Term Extraction (ATE)** — Identifies opinion targets in a sentence (e.g., *"pasta"*, *"service"*)
2. **Aspect Sentiment Classification (ASC)** — Classifies the sentiment towards each extracted aspect (*positive*, *negative*, *neutral*, *conflict*)

Unlike pipeline approaches that treat these as separate tasks, our architecture performs **both tasks in a single forward pass**, leveraging shared representations for better accuracy and efficiency.

### Key Innovation

- **ModernBERT backbone** — Latest-generation encoder with RoPE positional embeddings, replacing legacy BERT
- **Relational Graph Attention (RGAT)** — Encodes syntactic dependency structure (nsubj, amod, obj, etc.) to capture long-range aspect-opinion relationships
- **Joint BIO + Sentiment heads** — Unified architecture with token-level aspect extraction and span-level sentiment classification

---

## Architecture

```
                           ┌──────────────────────┐
                           │   Raw Text Input      │
                           │ "The pasta was great" │
                           └──────────┬───────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
            ┌───────▼───────┐ ┌──────▼──────┐   ┌──────▼──────┐
            │  ModernBERT   │ │  SpaCy NLP  │   │  Tokenizer  │
            │  Tokenizer    │ │  Dep Parse  │   │  Alignment  │
            └───────┬───────┘ └──────┬──────┘   └──────┬──────┘
                    │                │                 │
                    │         ┌──────▼──────┐          │
                    │         │  Adjacency   │         │
                    │         │  Tensor [R]   │        │
                    │         └──────┬──────┘          │
                    │                │                 │
            ┌───────▼────────────────▼─────────────────▼──┐
            │                                             │
            │        ModernBERT Encoder (22 layers)       │
            │        ═══════════════════════════          │
            │        RoPE + Flash Attention               │
            │        (8 layers unfrozen for fine-tuning)  │
            │                                             │
            └──────────────────┬──────────────────────────┘
                               │
                        ┌──────▼──────┐
                        │    RGAT     │
                        │   Layer     │
                        │ (7relations)│
                        └──────┬──────┘
                               │
                    ┌──────────┼──────────┐
                    │                     │
            ┌───────▼───────┐    ┌────────▼────────┐
            │   ATE Head    │    │    ASC Head     │
            │  Token-level  │    │   Span-level    │
            │  BIO Tagger   │    │   Classifier    │
            │  ───────────  │    │   ────────────  │
            │  O / B-ASP /  │    │   pos / neg /   │
            │  I-ASP        │    │   neu / conflict│
            └───────┬───────┘    └────────┬────────┘
                    │                     │
                    ▼                     ▼
            Extracted Aspects      Sentiment Labels
            ["pasta", "service"]   [positive, negative]
```

**RGAT Relation Types:**

| Relation | Description | Example |
|----------|-------------|---------|
| `nsubj` | Nominal subject | *pasta* ← was |
| `amod` | Adjectival modifier | great → *pasta* |
| `obj` | Direct object | loved → *pasta* |
| `advmod` | Adverbial modifier | *terribly* → slow |
| `neg` | Negation | *not* → good |
| `compound` | Compound words | *butter* → chicken |
| `conj` | Conjunction | pasta *and* service |

---

## 📊 Results

### Per-Dataset Performance

| Dataset | ATE F1 (Strict) | ATE F1 (Partial) | ASC Accuracy | ASC Macro-F1 | Combined F1 |
|---------|:---------------:|:-----------------:|:------------:|:------------:|:-----------:|
| **SemEval 2014** | **78.35%** | **86.45%** | 76.10% | 56.73% | **67.54%** |
| **SemEval 2015** | 67.96% | 74.94% | 83.84% | 59.54% | 63.75% |
| **SemEval 2016** | 67.20% | 77.90% | **87.79%** | **65.28%** | 66.24% |

### Per-Class Sentiment F1

| Class | 2014 | 2015 | 2016 |
|-------|:----:|:----:|:----:|
| Positive | 87.88% | 90.08% | **92.58%** |
| Negative | 65.42% | 78.02% | **84.21%** |
| Neutral | **53.61%** | 10.53% | 19.05% |
| Conflict | **20.00%** | — | — |

### SOTA Comparison (SemEval 2014 Restaurant)

| Model | ATE F1 | ASC Accuracy | Year |
|-------|:------:|:------------:|:----:|
| BERT-SPC | — | 76.96% | 2019 |
| AEN-BERT | — | 76.31% | 2019 |
| LCF-BERT | — | 78.47% | 2019 |
| DualGCN-BERT | — | 75.92% | 2021 |
| **Ours (ModernBERT-RGAT)** | **78.35%** | **76.10%** | **2026** |

> **Note:** Most SOTA models perform ATE and ASC as separate tasks. Our model is one of few doing **joint extraction + classification** in a single pass, which is a harder but more practical task.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Backbone** | [ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) (149M params) |
| **Graph Network** | Relational Graph Attention Network (RGAT) |
| **Tokenization** | HuggingFace Tokenizers with BIO alignment |
| **Dependency Parse** | spaCy `en_core_web_sm` |
| **Training** | Mixed-precision (FP16), gradient accumulation, differential LR |
| **Loss** | Joint weighted CrossEntropy (ATE + ASC) with label smoothing |
| **Regularization** | Dropout (0.4), weight decay (0.05), gradient clipping |
| **Demo** | Gradio interactive web app |

---

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone https://github.com/karthik0306/SOTA-ModernBERT-RGAT-Joint-Aspect-Sentiment-Extraction-for-Food-Tech-Reviews.git
cd SOTA-ModernBERT-RGAT-Joint-Aspect-Sentiment-Extraction-for-Food-Tech-Reviews

# Create environment
conda create -n modernbert_rgat python=3.10 -y
conda activate modernbert_rgat

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run Inference

```python
from src.inference import load_predictor

# Load a trained model
predictor = load_predictor(year='2014', device='cpu')

# Predict
text = "The pasta was delicious but the service was terrible."
results = predictor.predict(text)

for r in results:
    print(f"  {r.aspect} → {r.sentiment} ({r.confidence:.0%})")
# Output:
#   pasta → positive (84%)
#   service → negative (91%)
```

### 3. Launch Demo App

```bash
python app.py
# Opens Gradio interface at http://localhost:7860
```

### 4. Train from Scratch

Run the notebooks in order:
```
notebooks/01_eda_profiling.ipynb        # Data exploration
notebooks/02_data_pipeline.ipynb        # Data preprocessing
notebooks/03_model_architecture.ipynb   # Model design & smoke tests
notebooks/04_training.ipynb             # Training (requires GPU)
notebooks/05_evaluation.ipynb           # Evaluation & benchmarking
notebooks/06_inference_demo.ipynb       # Interactive inference demo
```

---

## 📁 Project Structure

```
├── configs/
│   └── config.yaml                # All hyperparameters & settings
│
├── Data/
│   ├── Raw_Data/                  # SemEval XML files (2014, 2015, 2016)
│   └── Processed_Data/            # Cleaned CSVs
│
├── src/                           # Core source code
│   ├── model.py                   # ModernBERT-RGAT architecture
│   ├── dataset.py                 # Dataset & preprocessing (multi-aspect BIO)
│   ├── data_pipeline.py           # Data loading, splits, class weights
│   ├── trainer.py                 # Training loop, scheduling, checkpointing
│   ├── losses.py                  # Joint loss function (ATE + ASC)
│   ├── evaluator.py               # Metrics computation & benchmarking
│   └── inference.py               # Inference pipeline
│
├── notebooks/                     # Step-by-step notebooks (01–06)
├── checkpoints/                   # Saved model weights (.pt)
├── outputs/                       # Results, plots, logs
├── app.py                         # Gradio demo application
├── requirements.txt               # Python dependencies
└── docs/                          # Architecture & results documentation
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Backbone | `answerdotai/ModernBERT-base` |
| Max sequence length | 96 |
| Batch size (effective) | 16 (4 × 4 accumulation) |
| BERT learning rate | 1e-5 |
| Head learning rate | 3e-4 |
| Weight decay | 0.05 |
| Warmup | 10% of total steps |
| Label smoothing | 0.1 |
| Head dropout | 0.4 |
| Unfrozen BERT layers | 8 / 22 |
| Mixed precision | FP16 |
| Early stopping patience | 5 epochs |

---

## 📝 Notebooks Guide

| # | Notebook | Description |
|---|----------|-------------|
| 01 | **EDA & Data Profiling** | Comprehensive analysis of 3 SemEval datasets with NLP statistics |
| 02 | **Data Pipeline** | Tokenization, BIO labeling, adjacency tensors, DataLoader creation |
| 03 | **Model Architecture** | ModernBERT-RGAT design, parameter analysis, forward pass validation |
| 04 | **Training** | Full training with mixed-precision, early stopping, checkpointing |
| 05 | **Evaluation** | Metrics computation, confusion matrices, cross-dataset comparison |
| 06 | **Inference Demo** | 3-model comparison, visual highlighting, agreement analysis |

---

## 🔬 Key Design Decisions

1. **Multi-aspect BIO labeling** — Each training sample marks *all* aspects in the sentence (not just one), preventing conflicting targets for the ATE head when the same sentence appears multiple times.

2. **Differential learning rates** — BERT backbone fine-tuned at 1e-5, task heads at 3e-4 — prevents catastrophic forgetting while allowing heads to learn quickly.

3. **RGAT for syntax** — Dependency parse relations capture long-range aspect-opinion connections that positional encodings alone miss (e.g., "The pasta, which arrived late, was cold" — RGAT links *pasta* to *cold* via dependency path).

4. **Joint loss balancing** — Equal weighting (α=0.5) between ATE and ASC losses, both using standard CrossEntropyLoss to prevent magnitude imbalance.

---

## 📚 References

- **ModernBERT** — Warner et al., "Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference", 2024
- **RGAT** — Busbridge et al., "Relational Graph Attention Networks", 2019
- **SemEval ABSA** — Pontiki et al., "SemEval-2014/2015/2016 Task 4/12: Aspect Based Sentiment Analysis", 2014-2016
- **Joint ATE+ASC** — Li et al., "A Unified Model for Opinion Target Extraction and Target Sentiment Prediction", 2019

---

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built for advancing aspect-level sentiment analysis in food-tech**

</div>
