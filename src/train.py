"""
Training module for anomalous sound detection models.

This module provides functions to train different types of models including
autoencoders and classical ML approaches.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from torch.utils.data import DataLoader

from src.autoencoder import ConvAutoencoder, FCAutoencoder, LSTMAutoencoder
from src.dataloader import AudioDataset


logger = logging.getLogger(__name__)


def train_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-4,
    device: str = 'cuda',
    flatten: bool = True,
    patience: int = 5
) -> nn.Module:
    """
    Train an autoencoder model with early stopping.

    Args:
        model: PyTorch autoencoder model
        dataloader: DataLoader for training data
        num_epochs: Maximum number of training epochs
        lr: Learning rate
        device: Device to use ('cuda' or 'cpu')
        flatten: Whether to flatten input for FC models
        patience: Early stopping patience (epochs without improvement)

    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    logger.info(f"Training autoencoder for {num_epochs} epochs with lr={lr}")

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            x = x.to(device)

            optimizer.zero_grad()
            output = model(x)

            if flatten:
                loss = criterion(output, x.view(x.size(0), -1))
            else:
                loss = criterion(output, x)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            num_batches += x.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    logger.info("Training complete.")
    return model


def train_lstm_autoencoder(
    model: nn.Module,
    dataloader: DataLoader,
    num_epochs: int = 50,
    lr: float = 1e-3,
    device: str = 'cuda',
    patience: int = 5
) -> nn.Module:
    """
    Train LSTM autoencoder with sequence data preprocessing.

    Args:
        model: LSTM autoencoder model
        dataloader: DataLoader for training data
        num_epochs: Maximum number of training epochs
        lr: Learning rate
        device: Device to use
        patience: Early stopping patience

    Returns:
        Trained model
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    logger.info(f"Training LSTM autoencoder for {num_epochs} epochs with lr={lr}")

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in dataloader:
            mel_specs, _, _ = batch
            x = mel_specs.squeeze(1).permute(0, 2, 1).to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, x)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    logger.info("Training complete.")
    return model


def train_classical_ml(
    dataloader: DataLoader,
    model_type: str,
    config: dict
) -> object:
    """
    Train classical ML models (LOF, Isolation Forest).

    Args:
        dataloader: DataLoader for training data
        model_type: Type of model ('lof' or 'isolation_forest')
        config: Configuration dictionary

    Returns:
        Trained classical ML model
    """
    logger.info(f"Training {model_type} model")

    # Extract features
    vectors = []
    for batch in dataloader:
        specs, _, _ = batch
        for spec in specs:
            spec = spec.squeeze(0)
            vectors.append(spec.flatten().cpu().numpy())
    vectors = np.stack(vectors, axis=0)

    logger.info(f"Training on {vectors.shape[0]} samples with {vectors.shape[1]} features")

    # Train model
    if model_type == 'lof':
        model = LocalOutlierFactor(
            n_neighbors=config['classical_ml']['lof']['n_neighbors'],
            novelty=config['classical_ml']['lof']['novelty']
        )
        model.fit(vectors)
        logger.info(f"LOF model trained with n_neighbors={config['classical_ml']['lof']['n_neighbors']}")

    elif model_type == 'isolation_forest':
        model = IsolationForest(
            contamination=config['classical_ml']['isolation_forest']['contamination'],
            random_state=config['classical_ml']['isolation_forest']['random_state']
        )
        model.fit(vectors)
        logger.info(f"Isolation Forest trained with contamination={config['classical_ml']['isolation_forest']['contamination']}")

    else:
        raise ValueError(f"Unknown classical ML model type: {model_type}")

    return model


def train_model(config: dict, device: str):
    """
    Main training function that handles all model types.

    Args:
        config: Configuration dictionary
        device: Device to use for training

    Returns:
        Trained model
    """
    model_type = config['model']['type']
    logger.info(f"Starting training for {model_type}")

    # Setup data
    train_dir = Path(config['paths']['data_dir']) / config['paths']['train_subdir']

    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_mels=config['audio']['n_mels'],
        power=2.0,
        normalized=config['audio']['normalized']
    )

    train_dataset = AudioDataset(train_dir, transform=mel_transform)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )

    logger.info(f"Loaded {len(train_dataset)} training samples")

    # Get input dimensions
    sample_spec = train_dataset[0][0]
    input_dim = sample_spec.shape[-1] * sample_spec.shape[-2]

    # Train based on model type
    if model_type == 'fc_autoencoder':
        model = FCAutoencoder(
            input_dim=input_dim,
            bottleneck_dim=config['architecture']['fc_autoencoder']['bottleneck_dim']
        )
        model = train_autoencoder(
            model,
            train_dataloader,
            num_epochs=config['training']['num_epochs'],
            lr=config['training']['learning_rate'],
            device=device,
            flatten=True,
            patience=config['training']['patience']
        )

    elif model_type == 'conv_autoencoder':
        model = ConvAutoencoder(
            in_channels=1,
            enc_flat_dim=tuple(config['architecture']['conv_autoencoder']['enc_flat_dim'])
        )
        model = train_autoencoder(
            model,
            train_dataloader,
            num_epochs=config['training']['num_epochs'],
            lr=config['training']['learning_rate'],
            device=device,
            flatten=False,
            patience=config['training']['patience']
        )

    elif model_type == 'lstm_autoencoder':
        model = LSTMAutoencoder(
            input_size=config['audio']['n_mels'],
            hidden_size=config['architecture']['lstm_autoencoder']['hidden_size'],
            num_layers=config['architecture']['lstm_autoencoder']['num_layers']
        )
        model = train_lstm_autoencoder(
            model,
            train_dataloader,
            num_epochs=config['training']['num_epochs'],
            lr=config['training']['learning_rate'],
            device=device,
            patience=config['training']['patience']
        )

    elif model_type in ['lof', 'isolation_forest']:
        model = train_classical_ml(train_dataloader, model_type, config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Save model
    model_save_dir = Path(config['paths']['model_save_dir'])
    model_save_dir.mkdir(parents=True, exist_ok=True)

    if model_type in ['fc_autoencoder', 'conv_autoencoder', 'lstm_autoencoder']:
        save_path = model_save_dir / f"{model_type}.pth"
        torch.save(model.state_dict(), save_path)
    else:
        save_path = model_save_dir / f"{model_type}.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)

    logger.info(f"Model saved to {save_path}")

    return model
