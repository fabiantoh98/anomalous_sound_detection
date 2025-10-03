"""
Main entry point for Anomalous Sound Detection experiments.

This script provides a CLI for training and evaluating different anomaly detection models
on the DCASE 2023 challenge dataset.
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from src.train import train_model
from src.evaluate import evaluate_model


def setup_logging(log_level: str = 'INFO') -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config YAML file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Anomalous Sound Detection - DCASE 2023 Challenge'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='conf/config.yaml',
        help='Path to configuration file (default: conf/config.yaml)'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'both'],
        default='both',
        help='Run mode: train, evaluate, or both (default: both)'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['fc_autoencoder', 'conv_autoencoder', 'lstm_autoencoder', 'lof', 'isolation_forest'],
        help='Model type (overrides config file)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides config file)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config file)'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu'],
        help='Device to use (overrides config file)'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.model:
        config['model']['type'] = args.model
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.device:
        config['device'] = args.device

    # Set random seed
    set_seed(config['seed'])
    logger.info(f"Random seed set to {config['seed']}")

    # Set device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = 'cpu'
    logger.info(f"Using device: {device}")

    # Create necessary directories
    Path(config['paths']['model_save_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['results_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['paths']['figures_dir']).mkdir(parents=True, exist_ok=True)

    model_type = config['model']['type']
    logger.info(f"Model type: {model_type}")

    # Train model
    model = None
    if args.mode in ['train', 'both']:
        logger.info("=" * 50)
        logger.info("Starting training...")
        logger.info("=" * 50)
        model = train_model(config, device)
        logger.info("Training completed successfully")

    # Evaluate model
    if args.mode in ['evaluate', 'both']:
        logger.info("=" * 50)
        logger.info("Starting evaluation...")
        logger.info("=" * 50)
        results = evaluate_model(config, device, model)
        logger.info("Evaluation completed successfully")

        # Print summary
        logger.info("=" * 50)
        logger.info("RESULTS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Model: {model_type}")
        logger.info(f"AUC: {results['auc']:.4f}")
        logger.info(f"pAUC (max FPR={config['evaluation']['pauc_max_fpr']}): {results['pauc']:.4f}")
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Optimal Threshold: {results['optimal_threshold']:.6f}")
        logger.info("=" * 50)

    logger.info("Done!")


if __name__ == "__main__":
    main()
