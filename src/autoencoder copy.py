import torch
import torch.nn as nn

# 1. Fully Connected Autoencoder (requires flattening input)
class FCAutoencoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, bottleneck_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # flatten
        z = self.encoder(x)
        out = self.decoder(z)
        out = out.view(x.size(0), -1)  # reshape to original
        return out

# 2. Convolutional Autoencoder (for 2D spectrograms)
class ConvAutoencoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),  # [B, 16, H/2, W/2]
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),           # [B, 32, H/4, W/4]
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        # Crop output to match input size
        if out.shape != x.shape:
            min_freq = min(out.shape[2], x.shape[2])
            min_time = min(out.shape[3], x.shape[3])
            out = out[:, :, :min_freq, :min_time]
            x = x[:, :, :min_freq, :min_time]
        return out

# 3. LSTM Autoencoder (for treating spectrogram as sequence)
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, num_layers, batch_first=True)

    def forward(self, x):
        # x: [batch, seq_len, feature_dim]
        _, (h, _) = self.encoder(x)
        # Repeat hidden state for each time step
        repeated = h[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(repeated)
        return out