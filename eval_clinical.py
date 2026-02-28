import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

df = pd.read_csv("data/clinical/clinical_clean.csv")

model = xgb.XGBClassifier()
model.load_model("outputs/clinical/xgb_final.json")

X = df[TOP_FEATURES]
y_true = df["PCOS (Y/N)"]

y_prob = model.predict_proba(X)[:,1]
y_pred = [1 if p > 0.5 else 0 for p in y_prob]

print("Clinical Accuracy:", accuracy_score(y_true, y_pred))
print("Clinical Precision:", precision_score(y_true, y_pred))
print("Clinical Recall:", recall_score(y_true, y_pred))
print("Clinical F1:", f1_score(y_true, y_pred))
print("Clinical AUC:", roc_auc_score(y_true, y_prob))
