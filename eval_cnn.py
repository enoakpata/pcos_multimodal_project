import torch
import pandas as pd
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

y_true = []
y_prob = []

for _, row in df.iterrows():
    img = Image.open(row["path"]).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(img)).item()

    y_prob.append(prob)
    y_true.append(row["label"])

y_pred = [1 if p > 0.5 else 0 for p in y_prob]

print("CNN Accuracy:", accuracy_score(y_true, y_pred))
print("CNN Precision:", precision_score(y_true, y_pred))
print("CNN Recall:", recall_score(y_true, y_pred))
print("CNN F1:", f1_score(y_true, y_pred))
print("CNN AUC:", roc_auc_score(y_true, y_prob))
