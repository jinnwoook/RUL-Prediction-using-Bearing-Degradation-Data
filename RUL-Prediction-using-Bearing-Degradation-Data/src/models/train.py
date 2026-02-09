"""
Training utilities for bearing RUL prediction models.

This module provides:
- Dataset classes for loading preprocessed bearing data
- Training loop with validation
- Model evaluation utilities
"""

import os
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from .cnn_lstm import CNNLSTM


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 100
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    use_log_transform: bool = True
    device: str = "auto"


class BearingDataset(Dataset):
    """
    PyTorch Dataset for bearing vibration data.

    Loads preprocessed CSV files and prepares them for CNN-LSTM training.
    Supports normalization and log transformation of RUL targets.

    Attributes:
        X_list: List of input feature tensors
        y_list: List of RUL target tensors
        file_list: List of source filenames

    Example:
        >>> dataset = BearingDataset(root_dir="./data/preprocessed")
        >>> x, y = dataset[0]
        >>> print(x.shape, y.shape)
    """

    # Normalization parameters (from validation data)
    NORM_PARAMS = {
        "torque_mean": -6.08,
        "torque_std": 1.68,
        "temp1_mean": 102.80,
        "temp1_std": 12.30,
        "temp2_mean": 115.0,
        "temp2_std": 16.53
    }

    def __init__(
        self,
        root_dir: str,
        rul_values: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        log_transform_target: bool = True
    ):
        """
        Initialize the bearing dataset.

        Args:
            root_dir: Directory containing preprocessed CSV files
            rul_values: Dictionary mapping filenames to RUL values
            normalize: Whether to normalize input features
            log_transform_target: Whether to apply log1p to RUL targets
        """
        self.X_list: List[torch.Tensor] = []
        self.y_list: List[torch.Tensor] = []
        self.file_list: List[str] = []

        self.normalize = normalize
        self.log_transform_target = log_transform_target

        self._load_data(root_dir, rul_values)

    def _load_data(
        self,
        root_dir: str,
        rul_values: Optional[Dict[str, float]]
    ) -> None:
        """Load and process all CSV files from the directory."""
        folders = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ])

        for folder in tqdm(folders, desc="Loading data"):
            folder_path = os.path.join(root_dir, folder)
            csv_files = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith(".csv")
            ])

            for fname in csv_files:
                filepath = os.path.join(folder_path, fname)
                df = pd.read_csv(filepath)

                # Remove time column
                if "Time (s)" in df.columns:
                    df.drop(columns=["Time (s)"], inplace=True)

                # Normalize operational parameters
                if self.normalize:
                    df = self._normalize_features(df)

                # Convert to tensor
                x_tensor = torch.tensor(
                    df.values.astype('float32')
                ).contiguous()

                self.X_list.append(x_tensor)
                self.file_list.append(fname)

                # Add RUL target if provided
                if rul_values is not None and fname in rul_values:
                    rul = rul_values[fname]
                    if self.log_transform_target:
                        rul = np.log1p(rul)
                    self.y_list.append(torch.tensor([rul], dtype=torch.float32))

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply normalization to operational parameters."""
        params = self.NORM_PARAMS

        if 'Torque[Nm]' in df.columns:
            df['Torque[Nm]'] = (
                (df['Torque[Nm]'] - params["torque_mean"]) / params["torque_std"]
            )
        if 'TC SP Front[C]' in df.columns:
            df['TC SP Front[C]'] = (
                (df['TC SP Front[C]'] - params["temp1_mean"]) / params["temp1_std"]
            )
        if 'TC SP Rear[C]' in df.columns:
            df['TC SP Rear[C]'] = (
                (df['TC SP Rear[C]'] - params["temp2_mean"]) / params["temp2_std"]
            )

        return df

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        if self.y_list:
            return self.X_list[idx], self.y_list[idx]
        return self.X_list[idx], self.file_list[idx]


class TestDataset(Dataset):
    """
    Dataset for test/inference data without RUL labels.

    Used for generating predictions on validation or test sets.
    """

    NORM_PARAMS = BearingDataset.NORM_PARAMS

    def __init__(
        self,
        root_dir: str,
        normalize: bool = True
    ):
        """
        Initialize the test dataset.

        Args:
            root_dir: Directory containing preprocessed CSV files
            normalize: Whether to normalize input features
        """
        self.X_list: List[torch.Tensor] = []
        self.file_list: List[str] = []

        self._load_data(root_dir, normalize)

    def _load_data(self, root_dir: str, normalize: bool) -> None:
        """Load all CSV files for inference."""
        folders = sorted([
            f for f in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, f))
        ])

        for folder in tqdm(folders, desc="Loading test data"):
            folder_path = os.path.join(root_dir, folder)
            csv_files = sorted([
                f for f in os.listdir(folder_path)
                if f.endswith(".csv")
            ])

            for fname in csv_files:
                filepath = os.path.join(folder_path, fname)
                df = pd.read_csv(filepath)

                if "Time (s)" in df.columns:
                    df.drop(columns=["Time (s)"], inplace=True)

                if normalize:
                    params = self.NORM_PARAMS
                    if 'Torque[Nm]' in df.columns:
                        df['Torque[Nm]'] = (
                            (df['Torque[Nm]'] - params["torque_mean"]) /
                            params["torque_std"]
                        )
                    if 'TC SP Front[C]' in df.columns:
                        df['TC SP Front[C]'] = (
                            (df['TC SP Front[C]'] - params["temp1_mean"]) /
                            params["temp1_std"]
                        )
                    if 'TC SP Rear[C]' in df.columns:
                        df['TC SP Rear[C]'] = (
                            (df['TC SP Rear[C]'] - params["temp2_mean"]) /
                            params["temp2_std"]
                        )

                x_tensor = torch.tensor(
                    df.values.astype('float32')
                ).contiguous()

                self.X_list.append(x_tensor)
                self.file_list.append(fname)

    def __len__(self) -> int:
        return len(self.X_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        return self.X_list[idx], self.file_list[idx]


class Trainer:
    """
    Trainer class for CNN-LSTM model.

    Provides training loop with validation, early stopping,
    and model checkpointing.

    Attributes:
        model: CNN-LSTM model instance
        config: Training configuration
        device: Compute device (CPU/GPU)

    Example:
        >>> model = CNNLSTM(input_channels=15)
        >>> trainer = Trainer(model, config=TrainingConfig())
        >>> history = trainer.train(train_loader, val_loader)
    """

    def __init__(
        self,
        model: CNNLSTM,
        config: Optional[TrainingConfig] = None
    ):
        """
        Initialize the trainer.

        Args:
            model: CNN-LSTM model to train
            config: Training configuration
        """
        self.config = config or TrainingConfig()

        # Setup device
        if self.config.device == "auto":
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(self.config.device)

        self.model = model.to(self.device)

        # Initialize optimizer and loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        self.criterion = nn.MSELoss()

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": []
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            save_path: Path to save the best model

        Returns:
            Training history dictionary
        """
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss = self._train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validation phase
            val_loss = None
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader)
                self.history["val_loss"].append(val_loss)

                # Check for improvement
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0

                    if save_path:
                        torch.save(self.model.state_dict(), save_path)
                        print(f"Model saved to {save_path}")
                else:
                    self.patience_counter += 1

            # Log progress
            log_msg = f"Epoch {epoch+1}/{self.config.epochs} - Loss: {train_loss:.6f}"
            if val_loss is not None:
                log_msg += f" - Val Loss: {val_loss:.6f}"
            print(log_msg)

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        return self.history

    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            loss = self.criterion(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_epoch(self, val_loader: DataLoader) -> float:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)


def train_model(
    train_data_dir: str,
    rul_labels: Dict[str, float],
    model_save_path: str,
    config: Optional[TrainingConfig] = None
) -> Tuple[CNNLSTM, Dict[str, List[float]]]:
    """
    Train a CNN-LSTM model on bearing data.

    High-level function that handles data loading, model creation,
    and training.

    Args:
        train_data_dir: Directory with preprocessed training data
        rul_labels: Dictionary mapping filenames to RUL values
        model_save_path: Path to save the trained model
        config: Training configuration

    Returns:
        Tuple of (trained model, training history)
    """
    config = config or TrainingConfig()

    # Load dataset
    dataset = BearingDataset(
        root_dir=train_data_dir,
        rul_values=rul_labels,
        log_transform_target=config.use_log_transform
    )

    # Split into train/val
    val_size = int(len(dataset) * config.validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )

    # Get input channels from data
    sample_x, _ = dataset[0]
    input_channels = sample_x.shape[-1]

    # Create model
    model = CNNLSTM(input_channels=input_channels)

    # Train
    trainer = Trainer(model, config)
    history = trainer.train(train_loader, val_loader, model_save_path)

    return model, history


def evaluate_model(
    model: CNNLSTM,
    test_loader: DataLoader,
    use_log_transform: bool = True,
    device: Optional[torch.device] = None
) -> List[Tuple[str, float]]:
    """
    Evaluate model on test data.

    Args:
        model: Trained CNN-LSTM model
        test_loader: DataLoader for test data
        use_log_transform: Whether predictions are log-transformed
        device: Compute device

    Returns:
        List of (filename, predicted_rul) tuples
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    predictions: List[Tuple[str, float]] = []

    with torch.no_grad():
        for x_test, fname in test_loader:
            x_test = x_test.to(device)
            pred = model(x_test)

            if use_log_transform:
                # Convert from log space
                rul_sec = torch.expm1(pred).cpu().numpy().flatten()
            else:
                rul_sec = pred.cpu().numpy().flatten()

            for i, f in enumerate(fname):
                predictions.append((f, float(rul_sec[i])))
                print(f"{f} -> Predicted RUL: {rul_sec[i]:.2f} seconds")

    return predictions
