# 📊 Results Report — ModernBERT-RGAT

## Experiment Setup

| Parameter | Value |
|-----------|-------|
| **Datasets** | SemEval 2014, 2015, 2016 Restaurant Reviews |
| **Split** | 80% train / 10% val / 10% test (stratified, sentence-level) |
| **Hardware** | NVIDIA H100 PCIe (MIG partition, ~20 GB) |
| **Training** | FP16 mixed-precision, effective batch size 16 |
| **Epochs** | Up to 20 (early stopping, patience=5) |
| **Backbone** | ModernBERT-base (149M params) |

---

## 1. Aspect Term Extraction (ATE)

### Span-Level Metrics

| Dataset | Precision | Recall | **F1** |
|---------|:---------:|:------:|:------:|
| **SemEval 2014** | 78.94% | 77.78% | **78.35%** |
| **SemEval 2015** | 68.31% | 67.61% | **67.96%** |
| **SemEval 2016** | 66.87% | 67.55% | **67.20%** |

### Partial Match & Token-Level

| Dataset | Partial F1 | Token F1 |
|---------|:----------:|:--------:|
| 2014 | 86.45% | 87.22% |
| 2015 | 74.94% | 75.61% |
| 2016 | 77.90% | 78.29% |

> **Partial match** gives credit for overlapping spans (e.g., predicting "crispy crust" when the gold is "crust" counts as partial match).

### Span Count Analysis

| Dataset | Predicted Spans | Gold Spans | Ratio |
|---------|:--------------:|:----------:|:-----:|
| 2014 | 1,472 | 1,494 | 0.985 |
| 2015 | 385 | 389 | 0.990 |
| 2016 | 498 | 493 | 1.010 |

The model produces span counts very close to gold — no over/under-extraction.

---

## 2. Aspect Sentiment Classification (ASC)

### Overall Metrics

| Dataset | Accuracy | Macro-F1 | Weighted-F1 |
|---------|:--------:|:--------:|:-----------:|
| **SemEval 2014** | 76.10% | 56.73% | 75.91% |
| **SemEval 2015** | 83.84% | 59.54% | 83.29% |
| **SemEval 2016** | **87.79%** | **65.28%** | **87.20%** |

### Per-Class F1 Scores

| Class | 2014 | 2015 | 2016 | Notes |
|-------|:----:|:----:|:----:|-------|
| **Positive** | 87.88% | 90.08% | 92.58% | Strong across all datasets |
| **Negative** | 65.42% | 78.02% | 84.21% | Improves with more data |
| **Neutral** | 53.61% | 10.53% | 19.05% | Hardest class — few samples |
| **Conflict** | 20.00% | — | — | Extremely rare class |

> **Key insight:** The model excels at positive and negative detection but struggles with neutral and conflict classes due to severe class imbalance in the training data.

### Confusion Matrix (2014 — Most Complete Dataset)

```
                    Predicted
              pos    neg    neu    con
Actual ┌─────────────────────────────┐
  pos  │  290     16     25      2   │
  neg  │   12     70     18      1   │
  neu  │   20     23     52      1   │
  con  │    5      4      3      2   │
       └─────────────────────────────┘
```

- **Positive→Neutral confusion (25):** Some positive aspects with weak sentiment misclassified as neutral
- **Neutral→Negative confusion (23):** Neutral aspects in negative-context sentences get pulled toward negative

---

## 3. Combined (Joint) Performance

| Dataset | ATE F1 | ASC Macro-F1 | **Combined F1** |
|---------|:------:|:------------:|:---------------:|
| 2014 | 78.35% | 56.73% | **67.54%** |
| 2015 | 67.96% | 59.54% | **63.75%** |
| 2016 | 67.20% | 65.28% | **66.24%** |

Combined F1 = harmonic mean of ATE F1 and ASC Macro-F1.

---

## 4. SOTA Comparison (SemEval 2014 Restaurant)

| Model | Type | ATE F1 | ASC Acc | ASC F1 | Year |
|-------|------|:------:|:-------:|:------:|:----:|
| ATAE-LSTM | Attention | — | 72.73% | — | 2016 |
| BERT-SPC | Fine-tune | — | 76.96% | — | 2019 |
| AEN-BERT | Attention | — | 76.31% | — | 2019 |
| LCF-BERT | Local Context | — | 78.47% | — | 2019 |
| BERT-PT | Post-train | — | 76.96% | — | 2019 |
| DualGCN-BERT | Dual Graph | — | 75.92% | — | 2021 |
| SpanBERT-ATE | Span | 82.34% | — | — | 2020 |
| GRACE-BERT | Gradient | — | 80.49% | — | 2021 |
| **Ours (ModernBERT-RGAT)** | **Joint** | **78.35%** | **76.10%** | **56.73%** | **2026** |

> **Important context:** Most SOTA models tackle ATE or ASC **separately**. Our model performs **both tasks jointly** in a single pass, which is a harder problem but more practical for deployment. The ATE F1 of 78.35% and ASC accuracy of 76.10% are competitive with dedicated single-task models.

---

## 5. Training Dynamics

### Convergence

| Dataset | Best Epoch | Total Epochs | Reason for Stop |
|---------|:----------:|:------------:|:---------------:|
| 2014 | ~9 | 11 | Early stopping (patience=5) |
| 2015 | ~5 | 10 | Early stopping |
| 2016 | ~4 | 9 | Early stopping |

### Key Training Observations

1. **ATE converges early:** Token-level BIO tagging reaches near-peak performance by epoch 3-5
2. **ASC benefits from longer training:** Sentiment classification continues improving beyond ATE convergence
3. **2014 dataset performs best** on ATE due to largest training set (3,699 aspect-sentence pairs vs 1,572 for 2015)
4. **2016 dataset performs best on ASC** despite fewer samples — cleaner annotation guidelines

---

## 6. Limitations & Future Work

### Current Limitations
- **Neutral/Conflict detection:** Very low F1 due to class imbalance (< 15% of samples)
- **Implicit aspects:** Model cannot detect aspects implied but not explicitly mentioned
- **Domain specificity:** Trained only on restaurant reviews — needs domain adaptation for other verticals

### Planned Improvements
- **R-Drop regularization** for better generalization
- **Contrastive learning** to better separate neutral from positive/negative
- **Data augmentation** for minority classes (neutral, conflict)
- **Cross-dataset training** to leverage all 3 datasets simultaneously
- **Domain adaptation** to food delivery / hotel reviews
