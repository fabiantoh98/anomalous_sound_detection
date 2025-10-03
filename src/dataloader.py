"""
DataLoader module for audio datasets.

This module provides PyTorch Dataset classes for loading and processing audio files
for anomalous sound detection tasks.
"""

import os
from typing import Optional, Tuple, Callable

import torch
from torch.utils.data import Dataset
import torchaudio

# Set backend for Windows compatibility
try:
    torchaudio.set_audio_backend("soundfile")
except:
    pass  # Backend already set or not available


class AudioDataset(Dataset):
    """
    PyTorch Dataset for loading audio files with optional transformations.

    This dataset loads .wav audio files from a directory and applies optional
    transformations (e.g., mel-spectrogram conversion) for training and evaluation.

    Attributes:
        directory: Path to the directory containing audio files
        file_list: List of .wav files in the directory
        transform: Optional transformation to apply to the audio waveform

    Example:
        >>> mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000)
        >>> dataset = AudioDataset('./data/bearing/train', transform=mel_transform)
        >>> waveform, sample_rate, filename = dataset[0]
    """

    def __init__(self, directory: str, transform: Optional[Callable] = None) -> None:
        """
        Initialize the AudioDataset.

        Args:
            directory: Path to the directory containing .wav audio files
            transform: Optional callable to transform the waveform (e.g., MelSpectrogram)
        """
        self.directory = directory
        self.file_list = [f for f in os.listdir(directory) if f.endswith('.wav')]
        self.transform = transform

    def __len__(self) -> int:
        """
        Get the number of audio files in the dataset.

        Returns:
            Number of audio files
        """
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Load and process an audio file at the given index.

        Args:
            idx: Index of the audio file to load

        Returns:
            Tuple containing:
                - waveform: Audio waveform tensor (transformed if transform is provided)
                - sample_rate: Sample rate of the audio file
                - filename: Name of the audio file
        """
        file_path = os.path.join(self.directory, self.file_list[idx])
        waveform, sample_rate = torchaudio.load(file_path)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, sample_rate, self.file_list[idx]


# Legacy class names for backward compatibility
FanAudioDataset = AudioDataset
TrashAudioDataset = AudioDataset
