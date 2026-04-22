import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

from src.models.simple_cnn import SimpleCNN
from src.utils.constants import PATHOLOGIES


# -----------------------------
# KONFIGURASJON
# -----------------------------
MODEL_PATH = "best_simple_cnn_small.pth"
IMAGE_PATH = "test_image.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Må samsvare med modellen du trente
SIMPLE_CNN_CHANNELS = (16, 32, 64)


# -----------------------------
# LAST MODELL OG BILDE
# -----------------------------
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original = np.array(image)
    tensor = get_transform()(image).unsqueeze(0)
    return original, tensor


def load_model():
    model = SimpleCNN(channels=SIMPLE_CNN_CHANNELS)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_last_conv_layer(model):
    for layer in reversed(model.features):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise ValueError("Fant ikke noe Conv2d-lag i model.features")


# -----------------------------
# GRAD-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)              # shape: [1, num_classes]
        score = output[:, class_idx]                  # valgt klasse
        score.backward(retain_graph=True)

        gradients = self.gradients[0]                 # [C, H, W]
        activations = self.activations[0]             # [C, H, W]

        weights = gradients.mean(dim=(1, 2))          # [C]

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32).to(DEVICE)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)

        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam.cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


# -----------------------------
# VISUALISERING
# -----------------------------
def overlay_heatmap(original_image, cam, alpha=0.4):
    h, w, _ = original_image.shape

    cam_resized = cv2.resize(cam, (w, h))
    heatmap = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original_bgr = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(original_bgr, 1 - alpha, heatmap, alpha, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)


def show_results(original, overlay, probs, target_idx):
    top_indices = np.argsort(probs)[::-1][:5]

    print("\nTopp 5 prediksjoner:")
    for idx in top_indices:
        print(f"{PATHOLOGIES[idx]:30s}: {probs[idx]:.4f}")

    print(f"\nHeatmap laget for klasse: {PATHOLOGIES[target_idx]}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Originalbilde")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title(f"Grad-CAM: {PATHOLOGIES[target_idx]}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Fant ikke bilde: {IMAGE_PATH}")

    model = load_model()
    target_layer = get_last_conv_layer(model)

    gradcam = GradCAM(model, target_layer)

    original_image, input_tensor = load_image(IMAGE_PATH)
    input_tensor = input_tensor.to(DEVICE)

    with torch.enable_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).detach().cpu().numpy()[0]

    # Bruk topp-predikert klasse
    target_idx = int(np.argmax(probs))

    cam = gradcam.generate(input_tensor, target_idx)
    overlay = overlay_heatmap(original_image, cam)

    show_results(original_image, overlay, probs, target_idx)
    gradcam.remove_hooks()


if __name__ == "__main__":
    main()