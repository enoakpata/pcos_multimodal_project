import torch
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "data/images/clean_train_split.csv"
cnn_path = "outputs/cnn/cnn_final.pt"

df = pd.read_csv(csv_path)

model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)

ckpt = torch.load(cnn_path, map_location=device)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

probs = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    img = Image.open(row["path"]).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        p = torch.sigmoid(model(img)).item()

    probs.append(p)

df["p_img"] = probs
df[["p_img","label"]].to_csv("outputs/meta/image_probs.csv", index=False)

print("Saved image probabilities.")
