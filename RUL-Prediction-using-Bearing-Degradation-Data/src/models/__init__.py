"""
Model module for bearing RUL prediction.

This module provides:
- cnn_lstm: CNN-LSTM architecture for RUL regression
- train: Training utilities and data loading for model training
"""

from .cnn_lstm import CNNLSTM, CNNLSTMConfig
from .train import (
    BearingDataset,
    Trainer,
    train_model,
    evaluate_model,
)

__all__ = [
    "CNNLSTM",
    "CNNLSTMConfig",
    "BearingDataset",
    "Trainer",
    "train_model",
    "evaluate_model",
]
