"""
Autoencoder architectures for anomalous sound detection.

This module implements various autoencoder architectures including:
- Fully Connected Autoencoder (FCAutoencoder)
- Convolutional Autoencoder (ConvAutoencoder)
- LSTM Autoencoder (LSTMAutoencoder)
"""

import torch
import torch.nn as nn
from typing import Tuple


class FCAutoencoder(nn.Module):
    """
    Fully Connected Autoencoder for anomaly detection on flattened spectrograms.

    This autoencoder uses fully connected layers to compress and reconstruct
    flattened spectrogram representations. It's suitable for learning global
    patterns in audio features.

    Args:
        input_dim: Dimension of flattened input (e.g., n_mels * time_steps)
        bottleneck_dim: Dimension of the compressed latent representation (default: 512)

    Example:
        >>> model = FCAutoencoder(input_dim=40960, bottleneck_dim=512)
        >>> x = torch.randn(8, 40960)  # Batch of 8 flattened spectrograms
        >>> reconstruction = model(x)
        >>> reconstruction.shape
        torch.Size([8, 40960])
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 512) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2048),  # Reduced from 4096
            nn.ReLU(),
            nn.BatchNorm1d(2048),       # Added BatchNorm for stability
            nn.Dropout(0.3),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Dropout(0.3),
            nn.Linear(2048, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, input_dim) or (batch_size, ...)

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim)
        """
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(x.size(0), -1)
        return out


class ConvAutoencoder(nn.Module):
    """
    Convolutional Autoencoder for 2D spectrogram data.

    This autoencoder uses convolutional layers to preserve spatial structure
    in spectrograms, making it effective for capturing local patterns and features.

    Args:
        in_channels: Number of input channels (default: 1 for mono audio)
        enc_flat_dim: Shape after encoder (channels, height, width) for bottleneck

    Example:
        >>> model = ConvAutoencoder(in_channels=1, enc_flat_dim=(128, 8, 20))
        >>> x = torch.randn(8, 1, 128, 313)  # Batch of 8 mel-spectrograms
        >>> reconstruction = model(x)
    """

    def __init__(self, in_channels: int = 1, enc_flat_dim: Tuple[int, int, int] = (128, 8, 20)) -> None:
        super().__init__()
        self.c, self.h, self.w = enc_flat_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2),

            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout2d(0.2)
        )
        
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.c * self.h* self.w, 512),
            nn.ReLU(),
            nn.Linear(512, self.c * self.h * self.w),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),

            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the convolutional autoencoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Reconstructed tensor (cropped to match input dimensions if needed)
        """
        z = self.encoder(x)
        z_flat = self.bottleneck(z)
        z = z_flat.view(x.size(0), self.c , self.h, self.w)
        out = self.decoder(z)
        if out.shape != x.shape:
            min_freq = min(out.shape[2], x.shape[2])
            min_time = min(out.shape[3], x.shape[3])
            out = out[:, :, :min_freq, :min_time]
            x = x[:, :, :min_freq, :min_time]
        return out


class ConvAutoencoderv1(nn.Module):
    """
    Alternative Convolutional Autoencoder (v1) without bottleneck compression.

    This is a simpler variant that directly uses transposed convolutions
    without an intermediate fully connected bottleneck layer.

    Args:
        in_channels: Number of input channels (default: 1)

    Note:
        This is kept for backward compatibility. Use ConvAutoencoder for new projects.
    """

    def __init__(self, in_channels: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # [B, 16, 64, 313]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            # Second conv block
            nn.Conv2d(16, 32, 3, stride=2, padding=1),           # [B, 32, 32, 157]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv2d(32, 64, 3, stride=2, padding=1),           # [B, 64, 16, 79]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            # Bottleneck
            nn.Conv2d(64, 128, 3, stride=2, padding=1),          # [B, 128, 8, 40]
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            # Upsample from bottleneck
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the autoencoder.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width)

        Returns:
            Reconstructed tensor (cropped to match input dimensions)
        """
        z = self.encoder(x)
        out = self.decoder(z)
        # Crop output to match input size
        if out.shape != x.shape:
            min_freq = min(out.shape[2], x.shape[2])
            min_time = min(out.shape[3], x.shape[3])
            out = out[:, :, :min_freq, :min_time]
            x = x[:, :, :min_freq, :min_time]
        return out


class LSTMAutoencoder(nn.Module):
    """
    LSTM-based Autoencoder for sequential audio feature data.

    This autoencoder uses LSTM layers to capture temporal dependencies in
    sequential data like time-series spectrograms. It's particularly effective
    for learning patterns that evolve over time.

    Args:
        input_size: Size of input features at each time step (e.g., n_mels)
        hidden_size: Number of features in LSTM hidden state (default: 128)
        num_layers: Number of stacked LSTM layers (default: 2)

    Example:
        >>> model = LSTMAutoencoder(input_size=128, hidden_size=256, num_layers=2)
        >>> x = torch.randn(8, 313, 128)  # Batch of 8 sequences
        >>> reconstruction = model(x)
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        
        # Decoder LSTM
        self.decoder = nn.LSTM(
            hidden_size, input_size, num_layers, 
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        
        # Optional: Add a linear layer to compress the hidden state further
        self.bottleneck = nn.Linear(hidden_size, hidden_size // 2)
        self.expand = nn.Linear(hidden_size // 2, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM autoencoder.

        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            Reconstructed tensor of shape (batch_size, seq_len, feature_dim)
        """
        _, (h, c) = self.encoder(x)

        # Optional bottleneck compression
        compressed = self.bottleneck(h[-1])
        expanded = self.expand(compressed)

        # Repeat for all time steps
        repeated = expanded.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(repeated)
        return out