import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

from src.utils.constants import NUM_CLASSES


class ViTModel(nn.Module):
    def __init__(self):
        super(ViTModel, self).__init__()

        # Laster en ferdigtrent Vision Transformer
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

        # Bytter ut siste klassifikasjonslag slik at modellen passer CheXpert (14 klasser)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x):
        return self.model(x)