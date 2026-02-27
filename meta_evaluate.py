import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

img = pd.read_csv("outputs/meta/image_probs.csv")
clin = pd.read_csv("outputs/meta/clinical_probs.csv")

n = min(len(img), len(clin))

X = pd.DataFrame({
    "p_img": img["p_img"][:n],
    "p_clin": clin["p_clin"][:n]
})

y = clin["PCOS (Y/N)"][:n]

meta = joblib.load("outputs/meta/meta_model.pkl")

pred = meta.predict(X)
prob = meta.predict_proba(X)[:,1]

print("Accuracy:", accuracy_score(y, pred))
print("Precision:", precision_score(y, pred))
print("Recall:", recall_score(y, pred))
print("F1 Score:", f1_score(y, pred))
print("AUC:", roc_auc_score(y, prob))
