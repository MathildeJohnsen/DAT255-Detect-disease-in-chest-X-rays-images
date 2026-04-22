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
from PIL import Image

from src.data.chexpert_dataset import CheXpertDataset
from src.models.simple_cnn import SimpleCNN
from src.models.resnet_model import ResNet18Model
from src.models.vit_model import ViTModel
from src.utils.constants import NUM_CLASSES, PATHOLOGIES


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

TRAIN_CSV = "data/chexpert/train.csv"
DATA_ROOT = "data/chexpert"

TRAIN_SUBSET_SIZE = 20000
VAL_SUBSET_SIZE = 2000

BATCH_SIZE = 8
NUM_EPOCHS = 5
RANDOM_SEED = 42


# =========================
# HJELPEFUNKSJONER
# =========================

def custom_crop(img):
    """
    Enkel cropping for å redusere tekst, kanter og irrelevante områder.
    """
    width, height = img.size

    left = int(0.08 * width)
    right = int(0.92 * width)
    top = int(0.08 * height)
    bottom = int(0.88 * height)

    return img.crop((left, top, right, bottom))


def get_train_transform():
    return transforms.Compose([
        transforms.Lambda(custom_crop),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform():
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


def compute_pos_weights(dataset, num_classes):
    """
    Beregner klassevekter for multilabel BCE-loss.
    pos_weight = antall negative / antall positive
    """
    pos_counts = torch.zeros(num_classes, dtype=torch.float32)
    neg_counts = torch.zeros(num_classes, dtype=torch.float32)

    print("Beregner pos_weight fra train-datasettet...")

    for i in range(len(dataset)):
        _, labels, mask = dataset[i]

        labels = labels.float()
        mask = mask.float()

        pos_counts += labels * mask
        neg_counts += (1.0 - labels) * mask

        if i % 5000 == 0 and i > 0:
            print(f"  Prosessert {i}/{len(dataset)} eksempler")

    pos_weight = neg_counts / torch.clamp(pos_counts, min=1.0)
    return pos_weight


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch_idx, (images, labels, mask) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device).float()
        mask = mask.to(device).float()

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
            labels = labels.to(device).float()
            mask = mask.to(device).float()

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

        if len(np.unique(y_true)) < 2:
            print(f"{pathology:30}: N/A (kun én klasse i y_true)")
            continue

        auc = roc_auc_score(y_true, y_score)
        aucs.append(auc)
        print(f"{pathology:30}: {auc:.4f}")

    mean_auc = np.mean(aucs) if len(aucs) > 0 else 0.0
    return val_loss, mean_auc


def build_model(model_type, device, simple_cnn_variant="medium"):
    if model_type == "simple_cnn":
        channels = SIMPLE_CNN_CONFIGS[simple_cnn_variant]
        print(f"Kjører SimpleCNN ({simple_cnn_variant}) med channels={channels}")
        model = SimpleCNN(num_classes=NUM_CLASSES, channels=channels).to(device)
        learning_rate = 1e-3

    elif model_type == "resnet":
        print("Kjører ResNet18")
        model = ResNet18Model(NUM_CLASSES).to(device)
        learning_rate = 1e-4

    elif model_type == "vit":
        print("Kjører Vision Transformer (ViT-B/16)")
        model = ViTModel().to(device)
        learning_rate = 1e-4

    else:
        raise ValueError(f"Ukjent MODEL_TYPE: {model_type}")

    return model, learning_rate


# =========================
# MAIN
# =========================

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_transform = get_train_transform()
    val_transform = get_val_transform()

    print("Laster base-dataset uten transform for splitting...")
    base_dataset = CheXpertDataset(
        csv_path=TRAIN_CSV,
        data_root=DATA_ROOT,
        transform=None,
        uncertainty_policy="ignore"
    )

    print(f"Antall bilder i train.csv: {len(base_dataset)}")

    train_size = int(0.9 * len(base_dataset))
    val_size = len(base_dataset) - train_size

    generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_indices, val_indices = random_split(
        range(len(base_dataset)),
        [train_size, val_size],
        generator=generator
    )

    train_indices = train_indices.indices
    val_indices = val_indices.indices

    print(f"Train split: {len(train_indices)}")
    print(f"Val split: {len(val_indices)}")

    # Egne dataset-objekter slik at train og val kan ha ulike transforms
    train_dataset_full = CheXpertDataset(
        csv_path=TRAIN_CSV,
        data_root=DATA_ROOT,
        transform=train_transform,
        uncertainty_policy="ignore"
    )

    val_dataset_full = CheXpertDataset(
        csv_path=TRAIN_CSV,
        data_root=DATA_ROOT,
        transform=val_transform,
        uncertainty_policy="ignore"
    )

    # Subset for raskere kjøring
    train_subset_indices = train_indices[:min(TRAIN_SUBSET_SIZE, len(train_indices))]
    val_subset_indices = val_indices[:min(VAL_SUBSET_SIZE, len(val_indices))]

    train_dataset = Subset(train_dataset_full, train_subset_indices)
    val_dataset = Subset(val_dataset_full, val_subset_indices)

    print(f"Train subset: {len(train_dataset)}")
    print(f"Val subset: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model, learning_rate = build_model(
        model_type=MODEL_TYPE,
        device=device,
        simple_cnn_variant=SIMPLE_CNN_VARIANT
    )

    print("Beregner klassevekter...")
    pos_weights = compute_pos_weights(train_dataset, NUM_CLASSES).to(device)
    print("Pos weights:", pos_weights)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights,
        reduction="none"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_auc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = evaluate(model, val_loader, criterion, device)

        print(f"\nTrain Loss:      {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Val Mean AUROC:  {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc

            if MODEL_TYPE == "simple_cnn":
                best_model_path = f"best_simple_cnn_{SIMPLE_CNN_VARIANT}.pth"
            else:
                best_model_path = f"best_{MODEL_TYPE}.pth"

            torch.save(model.state_dict(), best_model_path)
            print(f"Beste modell lagret som {best_model_path}")

    if MODEL_TYPE == "simple_cnn":
        last_model_path = f"last_simple_cnn_{SIMPLE_CNN_VARIANT}.pth"
    else:
        last_model_path = f"last_{MODEL_TYPE}.pth"

    torch.save(model.state_dict(), last_model_path)
    print(f"\nSiste modell lagret som {last_model_path}")


if __name__ == "__main__":
    main()