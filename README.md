# Anomalous Sound Detection

> First-shot Unsupervised Anomalous Sound Detection for Machine Condition Monitoring

This project implements and evaluates various anomaly detection approaches for the [DCASE 2023 Challenge Task 2](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring). The goal is to detect anomalous sounds in industrial machinery using only normal sound samples for training.

## Overview

This implementation compares multiple approaches to anomalous sound detection:

**Deep Learning Autoencoders:**
- Fully Connected Autoencoder (FC-AE)
- Convolutional Autoencoder (Conv-AE)
- LSTM Autoencoder (LSTM-AE)

**Classical ML Methods:**
- Local Outlier Factor (LOF)
- Isolation Forest

## Key Features

- **Production-ready code** with proper logging, type hints, and docstrings
- **Flexible configuration** using YAML files
- **CLI interface** for easy experimentation
- **Comprehensive evaluation** including AUC, pAUC, and detailed metrics
- **Automated visualization** of results (ROC curves, confusion matrices)
- **Early stopping** to prevent overfitting
- **Reproducible** experiments with seed control

## Project Structure

```
.
├── conf/
│   └── config.yaml           # Configuration file
├── data/
│   └── bearing/
│       ├── train/            # Normal sound samples
│       └── test/             # Normal + anomalous samples
├── models/                    # Saved trained models
├── results/
│   ├── figures/              # Generated plots
│   └── *.json                # Evaluation metrics
├── src/
│   ├── autoencoder.py        # Model architectures
│   ├── dataloader.py         # Dataset classes
│   ├── train.py              # Training functions
│   ├── evaluate.py           # Evaluation functions
│   └── utils.py              # Utility functions
├── notebooks/                 # Jupyter notebooks for experiments
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
└── README.md
```

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, but recommended)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Anomalous_Sound_Detection
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

## Usage

### Quick Start

Train and evaluate a Convolutional Autoencoder:
```bash
python main.py --model conv_autoencoder --epochs 50
```

### Command Line Arguments

```bash
python main.py [OPTIONS]
```

**Options:**
- `--config`: Path to config file (default: `conf/config.yaml`)
- `--mode`: Run mode - `train`, `evaluate`, or `both` (default: `both`)
- `--model`: Model type - `fc_autoencoder`, `conv_autoencoder`, `lstm_autoencoder`, `lof`, `isolation_forest`
- `--epochs`: Number of training epochs (overrides config)
- `--batch-size`: Batch size (overrides config)
- `--device`: Device to use - `cuda` or `cpu`
- `--log-level`: Logging level - `DEBUG`, `INFO`, `WARNING`, `ERROR`

### Examples

**Train only:**
```bash
python main.py --mode train --model lstm_autoencoder --epochs 100
```

**Evaluate existing model:**
```bash
python main.py --mode evaluate --model conv_autoencoder
```

**Train on CPU:**
```bash
python main.py --device cpu --model fc_autoencoder
```

**Train classical ML model:**
```bash
python main.py --model lof
```

### Configuration

Edit `conf/config.yaml` to customize:
- Audio processing parameters (sample rate, FFT size, mel bins)
- Model architectures
- Training hyperparameters
- Evaluation settings
- File paths

## Dataset

This project uses the DCASE 2023 Challenge dataset. Download it from:
https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring

Place the data in the `data/` directory following this structure:
```
data/
└── bearing/         # or other machine type
    ├── train/       # Normal samples only
    └── test/        # Normal + anomalous samples
```

## Evaluation Metrics

The system evaluates models using:

- **AUC** (Area Under ROC Curve): Overall classification performance
- **pAUC** (Partial AUC): AUC up to 10% FPR (DCASE official metric)
- **Accuracy**: Correct classification rate
- **Precision/Recall/F1**: Detailed classification metrics
- **Confusion Matrix**: Visual breakdown of predictions

Results are saved to:
- JSON files in `results/` directory
- Plots in `results/figures/` directory

## Results

### Bearing Dataset (Example)

| Model | AUC | pAUC | Accuracy |
|-------|-----|------|----------|
| LOF | 0.670 | - | 65.0% |
| Conv-AE | 0.570 | - | 57.0% |
| FC-AE | 0.520 | - | 53.0% |
| LSTM-AE | 0.550 | - | 55.0% |
| Isolation Forest | 0.515 | - | 50.0% |

*Note: These are preliminary results from limited training (1-50 epochs). Performance improves with proper hyperparameter tuning.*

## Key Insights

Based on our experiments:

1. **Classical ML can outperform deep learning** on small datasets with good feature engineering (mel-spectrograms)
2. **LOF** performed best, capturing local density variations effectively
3. **Deep models require more training** - results above are undertrained
4. **Feature representation matters** - mel-spectrograms provided better results than raw spectrograms
5. **Threshold selection is critical** - F1-optimal thresholds significantly impact performance

## Model Architectures

### Fully Connected Autoencoder
- Input: Flattened mel-spectrogram (128 × 313 → 40,960)
- Encoder: 40960 → 2048 → 1024 → 512 (bottleneck)
- Decoder: 512 → 1024 → 2048 → 40960
- Features: Batch normalization, dropout (0.3)

### Convolutional Autoencoder
- Input: 2D mel-spectrogram (1 × 128 × 313)
- Encoder: 4 conv layers (16→32→64→128 channels)
- Bottleneck: Fully connected compression to 512
- Decoder: 4 transposed conv layers (128→64→32→16→1)
- Features: Batch norm, LeakyReLU, dropout (0.2)

### LSTM Autoencoder
- Input: Sequential mel-spectrogram (313 timesteps × 128 features)
- Encoder: 2-layer bidirectional LSTM (hidden size: 256)
- Bottleneck: Linear compression (256 → 128)
- Decoder: 2-layer LSTM
- Features: Dropout between layers

## Future Improvements

- [ ] Implement Variational Autoencoder (VAE)
- [ ] Add data augmentation (time/frequency masking, mixup)
- [ ] Multi-machine type training for domain generalization
- [ ] Ensemble methods combining multiple models
- [ ] Attention mechanisms for LSTM models
- [ ] Hyperparameter optimization (Optuna/Ray Tune)
- [ ] Real-time inference API
- [ ] Docker containerization

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## References

1. [DCASE 2023 Challenge Task 2](https://dcase.community/challenge2023/task-first-shot-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring)
2. [Introduction to Autoencoders - DataCamp](https://www.datacamp.com/tutorial/introduction-to-autoencoders)
3. [Convolutional Autoencoders](https://github.com/ebrahimpichka/conv-autoencoder)
4. [Transposed Convolution](https://www.geeksforgeeks.org/machine-learning/what-is-transposed-convolutional-layer/)
5. [Applied Machine Learning](https://ff12.fastforwardlabs.com/)

## License

MIT License

## Author

Fabian Toh - AI Apprentice at AI Singapore
Jasper Chew - AI Apprentice at AI Singapore

## Acknowledgments

This project was developed as part of a machine learning coursework, focusing on practical application of autoencoder architectures for real-world anomaly detection tasks.
