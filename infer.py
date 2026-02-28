import torch
import torchvision.models as models
import torchvision.transforms as transforms
import xgboost as xgb
import pandas as pd
import joblib
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
    def __init__(self, cnn_path, xgb_path, meta_path):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CNN
        self.cnn = models.efficientnet_b0(weights=None)
        self.cnn.classifier[1] = torch.nn.Linear(self.cnn.classifier[1].in_features, 1)

        checkpoint = torch.load(cnn_path, map_location=self.device)
        self.cnn.load_state_dict(checkpoint["model_state"])
        self.cnn.to(self.device)
        self.cnn.eval()

        # XGBoost
        self.xgb = xgb.XGBClassifier()
        self.xgb.load_model(xgb_path)

        # Meta Learner
        self.meta = joblib.load(meta_path)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])

    def predict_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logit = self.cnn(img)
            logit = torch.clamp(logit, -10, 10)
            prob = torch.sigmoid(logit).item()

        return prob

    def predict_clinical(self, clinical_dict):
        X = pd.DataFrame([clinical_dict])[TOP_FEATURES]
        prob = self.xgb.predict_proba(X)[:,1][0]
        return float(prob)

    def predict_final(self, img_path, clinical_dict, threshold=0.5):
        p_img = self.predict_image(img_path)
        p_clin = self.predict_clinical(clinical_dict)

        # META LEARNER DECISION
        X_meta = pd.DataFrame([[p_img, p_clin]], columns=["p_img","p_clin"])
        p_final = self.meta.predict_proba(X_meta)[:,1][0]

        label = int(p_final >= threshold)
        diagnosis = "PCOS Positive" if label==1 else "PCOS Negative"

        return {
            "p_img": p_img,
            "p_clin": p_clin,
            "final_probability": p_final,
            "label": label,
            "diagnosis": diagnosis
        }