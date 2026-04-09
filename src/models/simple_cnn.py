import torch
import torch.nn as nn
from src.utils.constants import NUM_CLASSES


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, channels=(16, 32, 64), dropout=0.3):
        super().__init__()

        layers = []
        in_channels = 3

        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)

        # Gjør output-størrelsen fast uansett hvor mange conv-lag du bruker
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x