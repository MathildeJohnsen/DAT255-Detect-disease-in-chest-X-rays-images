import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import random
import os

from src.models.simple_cnn import SimpleCNN
from src.models.resnet_model import ResNet18Model
from src.models.vit_model import ViTModel
from src.utils.constants import PATHOLOGIES


# =========================
# KONFIGURASJON
# =========================

MODEL_TYPE = "vit"   # "simple_cnn", "resnet", "vit"
SIMPLE_CNN_VARIANT = "large"   # "small", "medium", "large"

SIMPLE_CNN_CONFIGS = {
    "small": (16, 32),
    "medium": (16, 32, 64),
    "large": (32, 64, 128, 256)
}

# Sett riktig modellfil
if MODEL_TYPE == "simple_cnn":
    MODEL_PATH = f"best_simple_cnn_{SIMPLE_CNN_VARIANT}.pth"
elif MODEL_TYPE == "resnet":
    MODEL_PATH = "best_resnet.pth"
elif MODEL_TYPE == "vit":
    MODEL_PATH = "best_vit.pth"


CSV_PATH = "data/chexpert/valid.csv"


# =========================
# HJELPEFUNKSJONER
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def load_model(model_type, model_path, device, simple_cnn_variant="small"):
    if model_type == "simple_cnn":
        channels = SIMPLE_CNN_CONFIGS[simple_cnn_variant]
        model = SimpleCNN(channels=channels)

    elif model_type == "resnet":
     model = ResNet18Model(num_classes=len(PATHOLOGIES))

    elif model_type == "vit":
     model = ViTModel()

    else:
        raise ValueError(f"Ukjent MODEL_TYPE: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def get_random_image_path(csv_path):
    df = pd.read_csv(csv_path)

    # tilfeldig bilde
    random_idx = random.randint(0, len(df) - 1)
    image_path = df.iloc[random_idx, 0]

    # fikse path
    image_path = image_path.replace("\\", "/")

    # VELG EN av disse (basert på din mappe!)
    image_path = image_path.replace("CheXpert-v1.0-small/", "data/chexpert/")
    # image_path = image_path.replace("CheXpert-v1.0-small/", "")

    return image_path


def predict_image(model, image, device):
    transform = get_transform()

    if isinstance(image, str):
        image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        image = image.convert("RGB")
    else:
        raise ValueError("image må være filsti eller PIL.Image")

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    results = {
        pathology: float(prob)
        for pathology, prob in zip(PATHOLOGIES, probs)
    }

    return results


def get_top_predictions(results, top_k=5):
    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]


# =========================
# MAIN
# =========================

def main():
    device = get_device()
    print(f"Using device: {device}")

    if MODEL_TYPE == "simple_cnn":
     print(f"Modell: SimpleCNN ({SIMPLE_CNN_VARIANT}) | Fil: {MODEL_PATH}")
    elif MODEL_TYPE == "resnet":
     print(f"Modell: ResNet18 | Fil: {MODEL_PATH}")
    elif MODEL_TYPE == "vit":
     print(f"Modell: ViT | Fil: {MODEL_PATH}")

    model = load_model(
        model_type=MODEL_TYPE,
        model_path=MODEL_PATH,
        device=device,
        simple_cnn_variant=SIMPLE_CNN_VARIANT
    )

    image_path = get_random_image_path(CSV_PATH)

    print(f"\nBruker bilde: {image_path}")
    print(f"Finnes filen? {os.path.exists(image_path)}")

    if not os.path.exists(image_path):
        print("\n FEIL: Bildet finnes ikke. Sjekk path-fix i koden.")
        return

    results = predict_image(model, image_path, device)

    print("\nAlle prediksjoner:")
    for pathology, prob in results.items():
        print(f"{pathology:30s}: {prob:.4f}")

    print("\nTopp 5 prediksjoner:")
    for pathology, prob in get_top_predictions(results, top_k=5):
        print(f"{pathology:30s}: {prob:.4f}")


if __name__ == "__main__":
    main()