import torch.nn as nn
from torchvision import models

class ResNet18Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.model = models.resnet18(pretrained=True)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
    
    