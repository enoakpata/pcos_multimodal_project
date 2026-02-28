from typing import Optional
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError


class PCOSDataset(Dataset):
    def __init__(self, df, transform=None, strict: bool = True):
        """
        df must contain columns: path, label
        strict=True: raise if an image can't be opened
        strict=False: return a black image if corrupted (keeps training running)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.strict = strict

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        label = int(self.df.loc[idx, "label"])

        try:
            img = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            if self.strict:
                raise e
            # fallback (rare) – keep training stable if a bad file slips in
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)
