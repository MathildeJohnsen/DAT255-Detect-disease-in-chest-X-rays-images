# denne skal:
# laste valid- datasett
# laste en lagret modell
# beregne validation loss
# senere kunne regne metrics
# brukes for sammenligning mellom modeller

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.data.chexpert_dataset import CheXpertDataset
from src.models.simple_cnn import SimpleCNN
from src.models.resnet_model import ResNet18Model
from src.utils.constants import NUM_CLASSES, PATHOLOGIES

MODEL_TYPE = "simple_cnn"   # "simple_cnn" eller "resnet"


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_probs = []
    all_labels = []
    all_masks = []

    with torch.no_grad():
        for batch_idx, (images, labels, mask) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            outputs = model(images)

            loss_matrix = criterion(outputs, labels)
            masked_loss = loss_matrix * mask

            if mask.sum() > 0:
                loss = masked_loss.sum() / mask.sum()
            else:
                continue

            running_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_masks.append(mask.cpu())

            if batch_idx % 20 == 0:
                print(f"Val Batch {batch_idx}: loss = {loss.item():.4f}")

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_masks = torch.cat(all_masks, dim=0).numpy()

    auc_scores = {}

    for i in range(all_labels.shape[1]):
        valid_idx = all_masks[:, i] == 1
        y_true = all_labels[valid_idx, i]
        y_score = all_probs[valid_idx, i]

        if len(np.unique(y_true)) < 2:
            auc_scores[i] = None
        else:
            auc_scores[i] = roc_auc_score(y_true, y_score)

    valid_aucs = [score for score in auc_scores.values() if score is not None]
    mean_auc = np.mean(valid_aucs) if valid_aucs else None

    return avg_loss, auc_scores, mean_auc


def get_model_and_weights(device):
    if MODEL_TYPE == "simple_cnn":
        model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
        weights_path = "best_simple_cnn.pth"

    elif MODEL_TYPE == "resnet":
        model = ResNet18Model(num_classes=NUM_CLASSES).to(device)
        weights_path = "best_resnet.pth"

    else:
        raise ValueError("Ugyldig MODEL_TYPE. Bruk 'simple_cnn' eller 'resnet'.")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"Modell lastet fra {weights_path}")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Evaluerer modelltype: {MODEL_TYPE}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Starter å laste offisielt validation-dataset...")
    val_dataset = CheXpertDataset(
        csv_path="data/chexpert/valid.csv",
        data_root="data/chexpert",
        transform=transform,
        uncertainty_policy="ignore"
    )
    print(f"Validation-dataset lastet. Antall eksempler: {len(val_dataset)}")

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    print("Dataloader klar!")

    model = get_model_and_weights(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")

    print("Starter evaluering...")
    val_loss, auc_scores, mean_auc = evaluate(model, val_loader, criterion, device)

    print(f"\nValidation Loss: {val_loss:.4f}")

    print("\nAUROC per klasse:")
    for i, auc in auc_scores.items():
        class_name = PATHOLOGIES[i]
        if auc is None:
            print(f"{class_name:<30}: N/A")
        else:
            print(f"{class_name:<30}: {auc:.4f}")

    if mean_auc is not None:
        print(f"\nMean AUROC: {mean_auc:.4f}")
    else:
        print("\nMean AUROC: Kunne ikke beregnes")


if __name__ == "__main__":
    main()