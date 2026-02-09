"""
CNN-LSTM model for bearing Remaining Useful Life (RUL) prediction.

This module implements a hybrid CNN-LSTM architecture that combines:
- 1D CNN for local temporal feature extraction
- LSTM for sequential pattern learning
- Fully connected layers for RUL regression

The model is designed for multivariate time series input from
bearing vibration sensors and operational parameters.
"""

from typing import Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CNNLSTMConfig:
    """Configuration for CNN-LSTM model architecture."""
    input_channels: int = 15
    cnn_out_channels: int = 64
    cnn_kernel_size: int = 5
    pool_kernel_size: int = 2
    lstm_hidden_size: int = 64
    fc_hidden_size: int = 32
    dropout: float = 0.0


class CNNLSTM(nn.Module):
    """
    CNN-LSTM hybrid model for RUL prediction.

    Architecture:
    1. Conv1D: Extracts local features from time series
       - Captures short-term patterns like sudden spikes in vibration
    2. MaxPool1d: Reduces temporal dimension while retaining key features
    3. LSTM: Learns temporal dependencies and trends
       - Captures progression of bearing degradation over time
    4. FC Layers: Maps LSTM output to RUL value

    Input shape: (batch_size, time_steps, channels)
    Output shape: (batch_size, 1)

    Attributes:
        conv1: 1D convolutional layer for local feature extraction
        pool: Max pooling layer for downsampling
        lstm: LSTM layer for sequential modeling
        fc: Fully connected layers for regression

    Example:
        >>> model = CNNLSTM(input_channels=15)
        >>> x = torch.randn(32, 256000, 15)  # batch, time, features
        >>> output = model(x)
        >>> print(output.shape)
        torch.Size([32, 1])
    """

    def __init__(
        self,
        input_channels: int = 15,
        config: Optional[CNNLSTMConfig] = None
    ):
        """
        Initialize the CNN-LSTM model.

        Args:
            input_channels: Number of input channels/features
            config: Model configuration (default: None, uses defaults)
        """
        super(CNNLSTM, self).__init__()

        if config is None:
            config = CNNLSTMConfig(input_channels=input_channels)

        self.config = config

        # Conv1D: Extract local temporal features
        # Detects sudden changes and patterns in vibration signals
        self.conv1 = nn.Conv1d(
            in_channels=config.input_channels,
            out_channels=config.cnn_out_channels,
            kernel_size=config.cnn_kernel_size
        )

        # MaxPooling: Reduce sequence length while keeping important features
        self.pool = nn.MaxPool1d(kernel_size=config.pool_kernel_size)

        # LSTM: Learn temporal dependencies
        # Captures degradation trends over time
        self.lstm = nn.LSTM(
            input_size=config.cnn_out_channels,
            hidden_size=config.lstm_hidden_size,
            batch_first=True
        )

        # FC layers: Map to RUL prediction
        self.fc = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, config.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(config.fc_hidden_size, 1)  # RUL regression output
        )

        if config.dropout > 0:
            self.dropout = nn.Dropout(config.dropout)
        else:
            self.dropout = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the CNN-LSTM model.

        Args:
            x: Input tensor of shape (batch, time_steps, channels)

        Returns:
            RUL prediction tensor of shape (batch, 1)
        """
        # x: (batch, time_steps, channels)

        # Permute for Conv1D: (batch, channels, time_steps)
        x = x.permute(0, 2, 1).contiguous()

        # Conv + ReLU
        x = F.relu(self.conv1(x))

        # Pooling
        x = self.pool(x)

        # Permute back for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1).contiguous()

        # LSTM forward pass
        x, (h_n, c_n) = self.lstm(x)

        # Use last timestep hidden state
        x = x[:, -1, :]

        # Optional dropout
        if self.dropout is not None:
            x = self.dropout(x)

        # FC layers for regression
        return self.fc(x)

    def get_num_parameters(self) -> int:
        """
        Get the total number of trainable parameters.

        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self) -> str:
        """
        Get a summary string of the model architecture.

        Returns:
            Model architecture summary string
        """
        return (
            f"CNNLSTM Model Summary:\n"
            f"  Input channels: {self.config.input_channels}\n"
            f"  CNN output channels: {self.config.cnn_out_channels}\n"
            f"  CNN kernel size: {self.config.cnn_kernel_size}\n"
            f"  Pool kernel size: {self.config.pool_kernel_size}\n"
            f"  LSTM hidden size: {self.config.lstm_hidden_size}\n"
            f"  FC hidden size: {self.config.fc_hidden_size}\n"
            f"  Total parameters: {self.get_num_parameters():,}\n"
        )


class CNNLSTMWithAttention(nn.Module):
    """
    CNN-LSTM with attention mechanism for improved RUL prediction.

    Adds a temporal attention layer to focus on the most relevant
    time steps for degradation prediction.

    This variant may provide better performance when certain time
    periods are more informative for RUL estimation.
    """

    def __init__(
        self,
        input_channels: int = 15,
        config: Optional[CNNLSTMConfig] = None
    ):
        """
        Initialize the CNN-LSTM model with attention.

        Args:
            input_channels: Number of input channels/features
            config: Model configuration
        """
        super(CNNLSTMWithAttention, self).__init__()

        if config is None:
            config = CNNLSTMConfig(input_channels=input_channels)

        self.config = config

        self.conv1 = nn.Conv1d(
            in_channels=config.input_channels,
            out_channels=config.cnn_out_channels,
            kernel_size=config.cnn_kernel_size
        )
        self.pool = nn.MaxPool1d(kernel_size=config.pool_kernel_size)

        self.lstm = nn.LSTM(
            input_size=config.cnn_out_channels,
            hidden_size=config.lstm_hidden_size,
            batch_first=True
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size),
            nn.Tanh(),
            nn.Linear(config.lstm_hidden_size, 1)
        )

        self.fc = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, config.fc_hidden_size),
            nn.ReLU(),
            nn.Linear(config.fc_hidden_size, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention mechanism.

        Args:
            x: Input tensor of shape (batch, time_steps, channels)

        Returns:
            RUL prediction tensor of shape (batch, 1)
        """
        # CNN feature extraction
        x = x.permute(0, 2, 1).contiguous()
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1).contiguous()

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Attention weights
        attention_weights = self.attention(lstm_out)
        attention_weights = F.softmax(attention_weights, dim=1)

        # Weighted sum of LSTM outputs
        context = torch.sum(attention_weights * lstm_out, dim=1)

        # FC layers
        return self.fc(context)


def load_model(
    model_path: str,
    input_channels: int = 15,
    device: Optional[torch.device] = None
) -> CNNLSTM:
    """
    Load a pretrained CNN-LSTM model.

    Args:
        model_path: Path to the saved model weights
        input_channels: Number of input channels
        device: Device to load the model on (default: auto-detect)

    Returns:
        Loaded CNNLSTM model in evaluation mode

    Example:
        >>> model = load_model("weights/best_model_0.66.pth")
        >>> model.eval()
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNNLSTM(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model
