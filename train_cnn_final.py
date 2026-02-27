import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["path"]).convert("RGB")
        label = torch.tensor(row["label"], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, label

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = "data/images/clean_train_split.csv"
    out_path = "outputs/cnn/cnn_final.pt"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(csv_path, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    EPOCHS = 5  # keep small for time

    print("Training final CNN...")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(loader):.4f}")

    torch.save(
        {"model_state": model.state_dict()},
        out_path
    )

    print("Final CNN model saved to:", out_path)

if __name__ == "__main__":
    main()
