from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from src.utils.constants import PATHOLOGIES


class CheXpertDataset(Dataset):
    def __init__(self, csv_path, data_root, transform=None, uncertainty_policy="ignore"):
        self.df = pd.read_csv(csv_path)
        self.data_root = Path(data_root)
        self.transform = transform
        self.uncertainty_policy = uncertainty_policy

        # Antar at første kolonne i CSV er Path
        self.path_col = self.df.columns[0]
        self.label_cols = PATHOLOGIES

        missing_cols = [col for col in self.label_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Mangler kolonner i CSV: {missing_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Fjern prefikset hvis det finnes i CSV-filen
        rel_path = str(row[self.path_col]).replace("CheXpert-v1.0-small/", "")
        img_path = self.data_root / rel_path

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = []
        mask = []

        for col in self.label_cols:
            value = row[col]

            if pd.isna(value):
                labels.append(0.0)
                mask.append(0.0)

            elif value == -1:
                if self.uncertainty_policy == "ignore":
                    labels.append(0.0)
                    mask.append(0.0)
                else:
                    labels.append(float(value))
                    mask.append(1.0)

            else:
                labels.append(float(value))
                mask.append(1.0)

        labels = torch.tensor(labels, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.float32)

        return image, labels, mask

