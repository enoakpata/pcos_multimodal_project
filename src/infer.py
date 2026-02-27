import torch
import torchvision.models as models
import torchvision.transforms as transforms
import xgboost as xgb
import pandas as pd
import numpy as np
from PIL import Image

TOP_FEATURES = [
    "Follicle No. (R)",
    "Weight gain(Y/N)",
    "hair growth(Y/N)",
    "Skin darkening (Y/N)",
    "Follicle No. (L)",
    "Cycle(R/I)",
    "Pimples(Y/N)",
    "AMH(ng/mL)"
]

class PCOSInference:
    def __init__(self, cnn_path, xgb_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CNN
        self.cnn = models.efficientnet_b0(weights=None)
        self.cnn.classifier[1] = torch.nn.Linear(self.cnn.classifier[1].in_features, 1)

        checkpoint = torch.load(cnn_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint["model_state"])
        self.cnn.to(self.device)
        self.cnn.eval()

        # Load XGBoost
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(xgb_path)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.cnn(img_tensor)
            prob = torch.sigmoid(logit).item()

        return prob

    def predict_clinical(self, feature_dict):
        X = pd.DataFrame([feature_dict])[TOP_FEATURES]
        prob = self.xgb.predict_proba(X)[:, 1][0]
        return float(prob)

    def fuse(self, p_img, p_clin, alpha=0.75):
        return alpha * p_img + (1 - alpha) * p_clin

    def classify(self, prob, threshold=0.5):
        label = int(prob >= threshold)
        diagnosis = "PCOS Positive" if label == 1 else "PCOS Negative"
        return label, diagnosis
    def predict_final(self, img_path, clinical_input, threshold=0.5):
        p_img = self.predict_image(img_path)
        p_clin = self.predict_clinical(clinical_input)
        p_fusion = self.fuse(p_img, p_clin)

        label, diagnosis = self.classify(p_fusion, threshold)

        return {
            "probability": p_fusion,
            "label": label,
            "diagnosis": diagnosis
        }      
