import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset

from src.data.chexpert_dataset import CheXpertDataset
from src.models.simple_cnn import SimpleCNN
from src.utils.constants import NUM_CLASSES


def main():
    # Velg device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Transformasjoner for bildene
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    print("Starter å laste dataset...")

    # Dataset
    dataset = CheXpertDataset(
        csv_path="data/chexpert/train.csv",
        data_root="data/chexpert",
        transform=transform,
        uncertainty_policy="ignore"
    )

    print(f"Fullt dataset lastet. Antall eksempler: {len(dataset)}")

    # Bruk bare et lite subset for testing/debugging
    dataset = Subset(dataset, range(50))
    print(f"Subset valgt. Antall eksempler: {len(dataset)}")

    print("Lager dataloader...")

    # Dataloader
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    print("Dataloader klar!")

    # Modell
    model = SimpleCNN(num_classes=NUM_CLASSES).to(device)
    print("Modell opprettet!")

    # Loss-funksjon
    criterion = nn.BCEWithLogitsLoss(reduction="none")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Antall epochs
    num_epochs = 5

    print("Starter trening...")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        num_batches = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        for batch_idx, (images, labels, mask) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            # Beregn loss per label
            loss_matrix = criterion(outputs, labels)

            # Bruk mask for å ignorere manglende/usikre labels
            masked_loss = loss_matrix * mask

            # Unngå deling på 0
            if mask.sum() > 0:
                loss = masked_loss.sum() / mask.sum()
            else:
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if batch_idx % 5 == 0:
                print(f"  Batch {batch_idx}: loss = {loss.item():.4f}")

        avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")

    # Lagre modellen
    torch.save(model.state_dict(), "simple_cnn.pth")
    print("Modell lagret som simple_cnn.pth")


if __name__ == "__main__":
    main()