"""
Quick test script to verify the setup is working correctly.
"""

import sys
from pathlib import Path

# Fix Windows encoding
if sys.platform == 'win32':
    import os
    os.environ['PYTHONIOENCODING'] = 'utf-8'

print("=" * 60)
print("Testing Anomalous Sound Detection Setup")
print("=" * 60)

# Test 1: Check imports
print("\n1. Testing imports...")
try:
    import torch
    import torchaudio
    import numpy as np
    import sklearn
    import yaml
    import matplotlib
    print("   [OK] All core libraries imported successfully")
except ImportError as e:
    print(f"   [FAIL] Import error: {e}")
    sys.exit(1)

# Test 2: Check project modules
print("\n2. Testing project modules...")
try:
    from src.autoencoder import FCAutoencoder, ConvAutoencoder, LSTMAutoencoder
    from src.dataloader import AudioDataset
    from src.train import train_model
    from src.evaluate import evaluate_model
    print("   [OK] All project modules imported successfully")
except ImportError as e:
    print(f"   [FAIL] Import error: {e}")
    sys.exit(1)

# Test 3: Check config file
print("\n3. Testing config file...")
try:
    with open('conf/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"   [OK] Config loaded: model={config['model']['type']}, epochs={config['training']['num_epochs']}")
except Exception as e:
    print(f"   [FAIL] Config error: {e}")
    sys.exit(1)

# Test 4: Check data directory
print("\n4. Testing data directory...")
data_dir = Path(config['paths']['data_dir'])
train_dir = data_dir / config['paths']['train_subdir']
test_dir = data_dir / config['paths']['test_subdir']

if train_dir.exists() and test_dir.exists():
    train_files = list(train_dir.glob('*.wav'))
    test_files = list(test_dir.glob('*.wav'))
    print(f"   [OK] Data found: {len(train_files)} train files, {len(test_files)} test files")
else:
    print(f"   [FAIL] Data directories not found")
    sys.exit(1)

# Test 5: Test model instantiation
print("\n5. Testing model instantiation...")
try:
    fc_model = FCAutoencoder(input_dim=40960, bottleneck_dim=512)
    conv_model = ConvAutoencoder(in_channels=1, enc_flat_dim=(128, 8, 20))
    lstm_model = LSTMAutoencoder(input_size=128, hidden_size=256, num_layers=2)
    print("   [OK] All models instantiated successfully")
except Exception as e:
    print(f"   [FAIL] Model instantiation error: {e}")
    sys.exit(1)

# Test 6: Check CUDA availability
print("\n6. Testing CUDA...")
if torch.cuda.is_available():
    print(f"   [OK] CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("   [WARN] CUDA not available, will use CPU")

# Test 7: Test dataset loading
print("\n7. Testing dataset loading...")
try:
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=config['audio']['sample_rate'],
        n_fft=config['audio']['n_fft'],
        hop_length=config['audio']['hop_length'],
        n_mels=config['audio']['n_mels'],
        power=2.0,
        normalized=config['audio']['normalized']
    )
    dataset = AudioDataset(train_dir, transform=mel_transform)
    sample = dataset[0]
    print(f"   [OK] Dataset loaded: {len(dataset)} samples, shape={sample[0].shape}")
except Exception as e:
    print(f"   [FAIL] Dataset loading error: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] All tests passed! Setup is ready.")
print("=" * 60)
print("\nYou can now run:")
print("  python main.py --model conv_autoencoder --epochs 50")
print("  python main.py --model lof")
print("  python main.py --help")
