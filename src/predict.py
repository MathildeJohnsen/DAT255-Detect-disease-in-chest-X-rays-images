import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
import time

from src.models.simple_cnn import SimpleCNN
from src.models.resnet_model import ResNet18Model
from src.models.vit_model import ViTModel
from src.utils.constants import PATHOLOGIES
from src.utils.gradcam import GradCAM, get_last_conv_layer, overlay_heatmap


# =========================
# KONFIGURASJON
# =========================

MODEL_TYPE = "simple_cnn"   # "simple_cnn", "resnet", "vit"
SIMPLE_CNN_VARIANT = "medium"   # "small", "medium", "large"

SIMPLE_CNN_CONFIGS = {
    "small": (16, 32),
    "medium": (16, 32, 64),
    "large": (32, 64, 128, 256)
}

if MODEL_TYPE == "simple_cnn":
    MODEL_PATH = f"best_simple_cnn_{SIMPLE_CNN_VARIANT}.pth"
elif MODEL_TYPE == "resnet":
    MODEL_PATH = "best_resnet.pth"
elif MODEL_TYPE == "vit":
    MODEL_PATH = "best_vit.pth"
else:
    raise ValueError(f"Ukjent MODEL_TYPE: {MODEL_TYPE}")

CSV_PATH = "data/chexpert/valid.csv"

# Sett til True hvis du vil vise heatmap for SimpleCNN
USE_HEATMAP = True


# =========================
# HJELPEFUNKSJONER
# =========================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_crop(img):
    width, height = img.size

    left = int(0.08 * width)
    right = int(0.92 * width)
    top = int(0.08 * height)
    bottom = int(0.88 * height)

    return img.crop((left, top, right, bottom))


def get_transform():
    return transforms.Compose([
        transforms.Lambda(custom_crop),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_model(model_type, model_path, device, simple_cnn_variant="small"):
    if model_type == "simple_cnn":
        channels = SIMPLE_CNN_CONFIGS[simple_cnn_variant]
        model = SimpleCNN(num_classes=len(PATHOLOGIES), channels=channels)

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

    random_idx = random.randint(0, len(df) - 1)
    image_path = df.iloc[random_idx, 0]

    image_path = image_path.replace("\\", "/")
    image_path = image_path.replace("CheXpert-v1.0-small/", "data/chexpert/")

    return image_path


def load_image(image):
    if isinstance(image, str):
        pil_image = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise ValueError("image må være filsti eller PIL.Image")

    return pil_image


def get_top_predictions(results, top_k=5):
    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:top_k]


def get_top_predictions_without_no_finding(results, top_k=5):
    filtered_results = {k: v for k, v in results.items() if k != "No Finding"}
    return sorted(filtered_results.items(), key=lambda x: x[1], reverse=True)[:top_k]


def predict_image(model, image, device):
    transform = get_transform()
    pil_image = load_image(image)

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        print("Raw outputs:", outputs.cpu().numpy())
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    results = {
        pathology: float(prob)
        for pathology, prob in zip(PATHOLOGIES, probs)
    }

    return results


def predict_image_with_heatmap(model, image, device):
    transform = get_transform()
    pil_image = load_image(image)

    # originalbilde for visning
    original_image = np.array(pil_image)

    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.enable_grad():
        outputs = model(image_tensor)
        probs = torch.sigmoid(outputs).detach().cpu().numpy()[0]

    results = {
        pathology: float(prob)
        for pathology, prob in zip(PATHOLOGIES, probs)
    }

    # Velg høyeste klasse uten "No Finding" for heatmap
    filtered_results = {k: v for k, v in results.items() if k != "No Finding"}

    if len(filtered_results) == 0:
        raise ValueError("Ingen gyldige klasser igjen etter filtrering av 'No Finding'.")

    top_predictions = get_top_predictions(filtered_results, top_k=5)
    target_pathology = top_predictions[0][0]
    target_idx = PATHOLOGIES.index(target_pathology)

    target_layer = get_last_conv_layer(model)
    gradcam = GradCAM(model, target_layer, device)
    cam = gradcam.generate(image_tensor, target_idx)
    heatmap_overlay = overlay_heatmap(original_image, cam)
    gradcam.remove_hooks()

    return results, original_image, heatmap_overlay, target_pathology


def show_heatmap(original_image, heatmap_overlay, target_pathology):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Originalbilde")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(heatmap_overlay)
    plt.title(f"Grad-CAM: {target_pathology}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


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
        print("\nFEIL: Bildet finnes ikke. Sjekk path-fix i koden.")
        return

    if MODEL_TYPE == "simple_cnn" and USE_HEATMAP:
        results, original_image, heatmap_overlay, target_pathology = predict_image_with_heatmap(
            model, image_path, device
        )
    else:
        results = predict_image(model, image_path, device)
        original_image = None
        heatmap_overlay = None
        target_pathology = None

    print("\nAlle prediksjoner:")
    for pathology, prob in results.items():
        print(f"{pathology:30s}: {prob:.4f}")

    print("\nTopp 5 prediksjoner:")
    for pathology, prob in get_top_predictions(results, top_k=5):
        print(f"{pathology:30s}: {prob:.4f}")

    print("\nTopp 5 prediksjoner (uten No Finding):")
    for pathology, prob in get_top_predictions_without_no_finding(results, top_k=5):
        print(f"{pathology:30s}: {prob:.4f}")

    if MODEL_TYPE == "simple_cnn" and USE_HEATMAP:
        print(f"\nHeatmap laget for: {target_pathology}")
        show_heatmap(original_image, heatmap_overlay, target_pathology)

        filename = f"gradcam_{target_pathology}_{int(time.time())}.png"
        heatmap_pil = Image.fromarray(heatmap_overlay)
        heatmap_pil.save(filename)
        print(f"Heatmap lagret som {filename}")


if __name__ == "__main__":
    main()