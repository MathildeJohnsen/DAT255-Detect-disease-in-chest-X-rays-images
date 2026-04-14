import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

from src.models.simple_cnn import SimpleCNN
from src.models.resnet_model import ResNet18Model
from src.utils.constants import PATHOLOGIES, NUM_CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Preprocessing (samme som trening)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------
# Last modeller
# -------------------------
def load_models():
    print("Laster modeller...")

    simple_model = SimpleCNN(channels=(16, 32, 64)).to(device)
    simple_model.load_state_dict(torch.load("best_simple_cnn_medium.pth", map_location=device))
    simple_model.to(DEVICE)
    simple_model.eval()

    resnet_model = ResNet18Model(num_classes=NUM_CLASSES)
    resnet_model.load_state_dict(torch.load("best_resnet.pth", map_location=DEVICE))
    resnet_model.to(DEVICE)
    resnet_model.eval()

    print("Modeller lastet!")

    return simple_model, resnet_model


simple_model, resnet_model = load_models()

# -------------------------
# Prediksjonsfunksjon
# -------------------------
def predict(model, image: Image.Image):
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

    results = {label: float(prob) for label, prob in zip(PATHOLOGIES, probs)}

    # sorter synkende
    results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))

    return results


# -------------------------
# Sammenlign begge modeller
# -------------------------
def compare_models(image):
    if image is None:
        return {}, {}

    simple_results = predict(simple_model, image)
    resnet_results = predict(resnet_model, image)

    # skjul "No Finding" hvis andre er høye
    def filter_results(results):
        filtered = {}
        for k, v in results.items():
            if k == "No Finding":
                continue
            if v > 0.5:   # terskel
                filtered[k] = v

        # hvis ingenting over terskel → vis original
        if len(filtered) == 0:
            return results
        return filtered

    simple_results = filter_results(simple_results)
    resnet_results = filter_results(resnet_results)

    return simple_results, resnet_results


# -------------------------
# Gradio UI
# -------------------------
demo = gr.Interface(
    fn=compare_models,
    inputs=gr.Image(type="pil", label="Last opp røntgenbilde"),
    outputs=[
        gr.Label(num_top_classes=5, label="SimpleCNN"),
        gr.Label(num_top_classes=5, label="ResNet"),
    ],
    title="Sammenligning av modeller: SimpleCNN vs ResNet",
    description=(
        "Last opp et røntgenbilde og sammenlign prediksjoner fra to modeller. "
        "Modellen returnerer sannsynlige funn (multi-label). "
        "Dette er kun en modellprediksjon og ikke en medisinsk diagnose."
    ),
)

# -------------------------
# Kjør app
# -------------------------
if __name__ == "__main__":
    demo.launch()