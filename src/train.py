# denne skal:
# laste train-datasett
# trene modellen
# skrive ut training loss og validation loss
# beregne AUROC
# lagre beste modell

import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import roc_auc_score

from src.data.chexpert_dataset import CheXpertDataset
from src.models.simple_cnn import SimpleCNN
from src.models.resnet_model import ResNet18Model
from src.models.vit_model import ViTModel
from src.utils.constants import NUM_CLASSES, PATHOLOGIES


# Velg modell:
MODEL_TYPE = "simple_cnn"   # "simple_cnn", "resnet" eller "vit"

# Brukes bare hvis MODEL_TYPE == "simple_cnn"
SIMPLE_CNN_VARIANT = "large"   # "small", "medium", "large"

SIMPLE_CNN_CONFIGS = {
    "small": (16, 32),
    "medium": (16, 32, 64),
    "large": (32, 64, 128, 256)
}


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels, mask) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss_matrix = criterion(outputs, labels)
        masked_loss = loss_matrix * mask

        if mask.sum() > 0:
            loss = masked_loss.sum() / mask.sum()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        if batch_idx % 100 == 0:
            if mask.sum() > 0:
                print(f"Train Batch {batch_idx}: loss = {loss.item():.4f}")

    return running_loss / num_batches if num_batches > 0 else 0.0


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
                running_loss += loss.item()
                num_batches += 1

            probs = torch.sigmoid(outputs)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    val_loss = running_loss / num_batches if num_batches > 0 else 0.0

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    aucs = []

    print("\nVal AUROC per klasse:")
    for i, pathology in enumerate(PATHOLOGIES):
        valid_idx = all_masks[:, i] == 1

        if np.sum(valid_idx) == 0:
            print(f"{pathology:30}: N/A (ingen gyldige labels)")
            continue

        y_true = all_labels[valid_idx, i]
        y_score = all_probs[valid_idx, i]

        # AUROC kan bare beregnes hvis både positive og negative finnes
        if len(np.unique(y_true)) < 2:
            print(f"{pathology:30}: N/A (kun én klasse i y_true)")
            continue

        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)
        print(f"{pathology:30}: {auc:.4f}")

    mean_auc = np.mean(aucs) if len(aucs) > 0 else 0.0

    return val_loss, mean_auc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Transform:
    # Vi bruker grayscale -> 3 kanaler fordi ResNet og ViT forventer RGB-lignende input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    print("Laster train-dataset...")
    dataset = CheXpertDataset(
            csv_path="data/chexpert/train.csv",
            data_root="data/chexpert",
            transform=transform,
            uncertainty_policy="ignore"
    )

    print(f"Antall bilder i train.csv: {len(dataset)}")

    # Splitter i train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Train split: {len(train_dataset)}")
    print(f"Val split: {len(val_dataset)}")

    # Subset for raskere og mer rettferdig sammenligning
    train_subset_size = 20000
    val_subset_size = 2000

    train_dataset = Subset(train_dataset, range(min(train_subset_size, len(train_dataset))))
    val_dataset = Subset(val_dataset, range(min(val_subset_size, len(val_dataset))))

    print(f"Train subset: {len(train_dataset)}")
    print(f"Val subset: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # Modellvalg
    if MODEL_TYPE == "simple_cnn":
        channels = SIMPLE_CNN_CONFIGS[SIMPLE_CNN_VARIANT]
        print(f"Kjører SimpleCNN ({SIMPLE_CNN_VARIANT}) med channels={channels}")
        model = SimpleCNN(num_classes=NUM_CLASSES, channels=channels).to(device)
        learning_rate = 1e-3

    elif MODEL_TYPE == "resnet":
        print("Kjører ResNet18")
        model = ResNet18Model(NUM_CLASSES).to(device)
        learning_rate = 1e-4

    elif MODEL_TYPE == "vit":
        print("Kjører Vision Transformer (ViT-B/16)")
        model = ViTModel().to(device)
        learning_rate = 1e-4

    else:
        raise ValueError(f"Ukjent MODEL_TYPE: {MODEL_TYPE}")

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    num_epochs = 5
    best_auc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"\nTrain Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val Mean AUROC: {val_auc:.4f}")

        # Lagrer beste modell
        if val_auc > best_auc:
            best_auc = val_auc

            if MODEL_TYPE == "simple_cnn":
                best_model_path = f"best_simple_cnn_{SIMPLE_CNN_VARIANT}.pth"
            else:
                best_model_path = f"best_{MODEL_TYPE}.pth"

            torch.save(model.state_dict(), best_model_path)
            print(f"Beste modell lagret som {best_model_path}")

    # Lagrer siste modell også
    if MODEL_TYPE == "simple_cnn":
        last_model_path = f"last_simple_cnn_{SIMPLE_CNN_VARIANT}.pth"
    else:
        last_model_path = f"last_{MODEL_TYPE}.pth"

    torch.save(model.state_dict(), last_model_path)
    print(f"\nSiste modell lagret som {last_model_path}")


if __name__ == "__main__":
    main()