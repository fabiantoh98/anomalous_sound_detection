"""
Evaluation module for anomalous sound detection models.

This module provides functions to evaluate trained models using various metrics
including AUC, pAUC, accuracy, precision, recall, and F1-score.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    auc,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
import torchaudio

from src.dataloader import AudioDataset
from src.autoencoder import FCAutoencoder, ConvAutoencoder, LSTMAutoencoder


logger = logging.getLogger(__name__)


def calculate_pauc(y_true: np.ndarray, y_scores: np.ndarray, max_fpr: float = 0.1) -> float:
    """
    Calculate partial AUC (pAUC) up to a maximum false positive rate.

    This is the official metric used in DCASE 2023 challenge.

    Args:
        y_true: True binary labels (0 or 1)
        y_scores: Predicted anomaly scores (higher = more anomalous)
        max_fpr: Maximum false positive rate to consider (default: 0.1)

    Returns:
        Partial AUC score normalized to [0, 1]
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)

    # Find the index where FPR exceeds max_fpr
    idx = np.where(fpr <= max_fpr)[0]

    if len(idx) == 0:
        return 0.0

    # Calculate partial AUC
    fpr_partial = fpr[idx]
    tpr_partial = tpr[idx]

    # Add the endpoint at max_fpr
    if fpr_partial[-1] < max_fpr:
        # Interpolate TPR at max_fpr
        tpr_at_max_fpr = np.interp(max_fpr, fpr, tpr)
        fpr_partial = np.append(fpr_partial, max_fpr)
        tpr_partial = np.append(tpr_partial, tpr_at_max_fpr)

    pauc = auc(fpr_partial, tpr_partial)

    # Normalize to [0, 1]
    pauc_normalized = pauc / max_fpr

    return pauc_normalized


def get_reconstruction_errors(
    model,
    dataloader: DataLoader,
    model_type: str,
    device: str
) -> np.ndarray:
    """
    Calculate reconstruction errors for all samples.

    Args:
        model: Trained model
        dataloader: DataLoader for test data
        model_type: Type of model ('fc_autoencoder', 'conv_autoencoder', 'lstm_autoencoder')
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Array of reconstruction errors (anomaly scores)
    """
    model.eval()
    errors = []

    with torch.no_grad():
        for batch in dataloader:
            specs, _, _ = batch

            if model_type == 'fc_autoencoder':
                x = specs.squeeze(1).view(specs.size(0), -1).to(device)
                recon = model(x)
                batch_errors = torch.mean((x - recon) ** 2, dim=1)

            elif model_type == 'conv_autoencoder':
                x = specs.to(device)
                recon = model(x)
                batch_errors = torch.mean((x - recon) ** 2, dim=(1, 2, 3))

            elif model_type == 'lstm_autoencoder':
                x = specs.squeeze(1).permute(0, 2, 1).to(device)
                recon = model(x)
                batch_errors = torch.mean((x - recon) ** 2, dim=(1, 2))

            errors.extend(batch_errors.cpu().numpy())

    return np.array(errors)


def get_labels_from_filenames(filenames: list) -> np.ndarray:
    """
    Extract labels from filenames.

    Args:
        filenames: List of audio filenames

    Returns:
        Array of labels (0=normal, 1=anomaly)
    """
    labels = []
    for fname in filenames:
        if "normal" in fname.lower():
            labels.append(0)
        elif "anomaly" in fname.lower():
            labels.append(1)
        else:
            labels.append(-1)  # Unknown
    return np.array(labels)


def find_optimal_threshold(
    precision: np.ndarray,
    recall: np.ndarray,
    thresholds: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal threshold based on specified metric.

    Args:
        precision: Array of precision values
        recall: Array of recall values
        thresholds: Array of threshold values
        metric: Metric to optimize ('f1', 'balanced', 'intersection')

    Returns:
        Tuple of (optimal_threshold, optimal_score)
    """
    if metric == 'f1':
        epsilon = 1e-8
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + epsilon)
        optimal_idx = np.argmax(f1_scores)
        return thresholds[optimal_idx], f1_scores[optimal_idx]

    elif metric == 'balanced':
        balanced_scores = (precision[:-1] + recall[:-1]) / 2
        optimal_idx = np.argmax(balanced_scores)
        return thresholds[optimal_idx], balanced_scores[optimal_idx]

    elif metric == 'intersection':
        diff = np.abs(precision[:-1] - recall[:-1])
        optimal_idx = np.argmin(diff)
        return thresholds[optimal_idx], (precision[optimal_idx] + recall[optimal_idx]) / 2

    else:
        raise ValueError(f"Unknown metric: {metric}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_scores: Predicted scores
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"ROC curve saved to {save_path}")
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    save_path: Optional[Path] = None
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
        save_path: Path to save the plot (optional)
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Normal', 'Anomaly'])
    ax.set_yticklabels(['Normal', 'Anomaly'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {model_name}')
    plt.colorbar(im, ax=ax)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
    plt.close()


def evaluate_model(config: dict, device: str, model=None) -> Dict[str, float]:
    """
    Evaluate trained model on test set.

    Args:
        config: Configuration dictionary
        device: Device to use ('cuda' or 'cpu')
        model: Pre-trained model (optional, will load from disk if None)

    Returns:
        Dictionary containing evaluation metrics
    """
    model_type = config['model']['type']

    # Setup data
    test_dir = Path(config['paths']['data_dir']) / config['paths']['test_subdir']

    # Create mel spectrogram transform
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_mels=config['audio']['n_mels'],
        power=2.0,
        normalized=config['audio']['normalized']
    )

    test_dataset = AudioDataset(test_dir, transform=mel_transform)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )

    # Get labels
    filenames = [test_dataset[i][2] for i in range(len(test_dataset))]
    test_labels = get_labels_from_filenames(filenames)
    valid_idx = test_labels != -1
    test_labels = test_labels[valid_idx]

    # Load or use provided model
    if model is None and model_type in ['fc_autoencoder', 'conv_autoencoder', 'lstm_autoencoder']:
        logger.info(f"Loading model from {config['paths']['model_save_dir']}")
        model = load_model(config, device)

    # Get predictions based on model type
    if model_type in ['fc_autoencoder', 'conv_autoencoder', 'lstm_autoencoder']:
        test_scores = get_reconstruction_errors(model, test_dataloader, model_type, device)
        test_scores = test_scores[valid_idx]

    elif model_type in ['lof', 'isolation_forest']:
        # For classical ML models, load and predict
        test_scores = evaluate_classical_model(config, test_dataloader, model_type)
        test_scores = test_scores[valid_idx]

    # Calculate metrics
    roc_auc = roc_auc_score(test_labels, test_scores)
    pauc = calculate_pauc(test_labels, test_scores, max_fpr=config['evaluation']['pauc_max_fpr'])

    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(test_labels, test_scores)
    optimal_threshold, _ = find_optimal_threshold(
        precision, recall, thresholds,
        metric=config['evaluation']['threshold_metric']
    )

    # Make predictions
    test_preds = (test_scores > optimal_threshold).astype(int)
    accuracy = np.mean(test_preds == test_labels)

    # Log results
    logger.info(f"AUC: {roc_auc:.4f}")
    logger.info(f"pAUC (max FPR={config['evaluation']['pauc_max_fpr']}): {pauc:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Optimal Threshold: {optimal_threshold:.6f}")

    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(test_labels, test_preds, target_names=['Normal', 'Anomaly']))

    # Save visualizations
    if config['visualization']['save_plots']:
        figures_dir = Path(config['paths']['figures_dir'])
        figures_dir.mkdir(parents=True, exist_ok=True)

        plot_roc_curve(
            test_labels, test_scores, model_type,
            save_path=figures_dir / f"{model_type}_roc_curve.{config['visualization']['plot_format']}"
        )

        plot_confusion_matrix(
            test_labels, test_preds, model_type,
            save_path=figures_dir / f"{model_type}_confusion_matrix.{config['visualization']['plot_format']}"
        )

    # Save results to JSON
    results = {
        'model_type': model_type,
        'auc': float(roc_auc),
        'pauc': float(pauc),
        'accuracy': float(accuracy),
        'optimal_threshold': float(optimal_threshold)
    }

    results_path = Path(config['paths']['results_dir']) / f"{model_type}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return results


def load_model(config: dict, device: str):
    """
    Load trained model from disk.

    Args:
        config: Configuration dictionary
        device: Device to use

    Returns:
        Loaded model
    """
    model_type = config['model']['type']
    model_path = Path(config['paths']['model_save_dir']) / f"{model_type}.pth"

    # Get input dimensions (simplified - you may need to adjust)
    if model_type == 'fc_autoencoder':
        input_dim = config['audio']['n_mels'] * 313  # Approximate time steps
        model = FCAutoencoder(
            input_dim=input_dim,
            bottleneck_dim=config['architecture']['fc_autoencoder']['bottleneck_dim']
        )
    elif model_type == 'conv_autoencoder':
        model = ConvAutoencoder(
            in_channels=1,
            enc_flat_dim=tuple(config['architecture']['conv_autoencoder']['enc_flat_dim'])
        )
    elif model_type == 'lstm_autoencoder':
        model = LSTMAutoencoder(
            input_size=config['audio']['n_mels'],
            hidden_size=config['architecture']['lstm_autoencoder']['hidden_size'],
            num_layers=config['architecture']['lstm_autoencoder']['num_layers']
        )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    logger.info(f"Model loaded from {model_path}")
    return model


def evaluate_classical_model(config: dict, test_dataloader: DataLoader, model_type: str) -> np.ndarray:
    """
    Evaluate classical ML models (LOF, Isolation Forest).

    Args:
        config: Configuration dictionary
        test_dataloader: DataLoader for test data
        model_type: Type of model ('lof' or 'isolation_forest')

    Returns:
        Array of anomaly scores
    """
    import pickle

    model_path = Path(config['paths']['model_save_dir']) / f"{model_type}.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Get test vectors
    test_vectors = []
    for batch in test_dataloader:
        specs, _, _ = batch
        for spec in specs:
            spec = spec.squeeze(0)
            test_vectors.append(spec.flatten().cpu().numpy())
    test_vectors = np.stack(test_vectors, axis=0)

    # Get anomaly scores
    if model_type == 'lof':
        scores = -model.decision_function(test_vectors)
    else:  # isolation_forest
        scores = -model.decision_function(test_vectors)

    return scores
