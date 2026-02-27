import pandas as pd
import xgboost as xgb

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
probs = model.predict_proba(X)[:,1]

df["p_clin"] = probs
df[["p_clin","PCOS (Y/N)"]].to_csv("outputs/meta/clinical_probs.csv", index=False)

print("Saved clinical probabilities.")
