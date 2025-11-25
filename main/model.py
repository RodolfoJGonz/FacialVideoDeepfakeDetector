import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50_LSTM(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=1, num_classes=2):
        super().__init__()

        # --- 1. CNN backbone (ResNet50) ---
        self.cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.cnn.fc = nn.Identity()         # remove classifier -> output dim = 2048

        self.feature_dim = 2048

        # --- 2. LSTM ---
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

        # --- 3. Classifier ---
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: [B, T, 3, H, W]
        """
        B, T, C, H, W = x.shape

        # merge batch & time
        x = x.view(B * T, C, H, W)

        # CNN feature extraction on each frame
        feats = self.cnn(x)                 # [B*T, 2048]

        # reshape back to time sequence
        feats = feats.view(B, T, self.feature_dim)

        # LSTM temporal modeling
        lstm_out, _ = self.lstm(feats)      # [B, T, hidden_dim]

        # take the LAST timestep output
        last_out = lstm_out[:, -1, :]       # [B, hidden_dim]

        logits = self.fc(last_out)          # [B, num_classes]
        return logits

