"""
ModernBERT-RGAT | Joint Aspect Extraction & Sentiment Classification
=====================================================================

A deep learning engine for aspect-level sentiment analysis using
ModernBERT with Relational Graph Attention Networks (RGAT).

Modules:
    model          - ModernBERT-RGAT architecture (RGAT + BIO tagger + sentiment head)
    dataset        - ABSAPreprocessor, ABSADataset (multi-aspect BIO labeling)
    data_pipeline  - Data loading, splits, class weights, DataLoader factory
    trainer        - Training loop with mixed-precision, scheduling, checkpointing
    losses         - Joint loss function (ATE CrossEntropy + ASC CrossEntropy)
    evaluator      - Metrics computation (ATE span F1, ASC accuracy/F1, confusion matrix)
    inference      - End-to-end inference pipeline with HTML visualization
"""

__version__ = "1.0.0"
__author__ = "Karthik Thota"
