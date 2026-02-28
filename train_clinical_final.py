import os
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

def main():
    project_dir = os.path.abspath(".")
    data_path = os.path.join(project_dir, "data/clinical/clinical_clean.csv")
    out_path = os.path.join(project_dir, "outputs/clinical/xgb_final.json")

    df = pd.read_csv(data_path)

    X = df[TOP_FEATURES]
    y = df["PCOS (Y/N)"]

    scale_pos_weight = (y == 0).sum() / (y == 1).sum()

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.05,
        max_depth=4,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    model.fit(X, y)

    model.save_model(out_path)
    print("Final clinical model saved to:", out_path)

if __name__ == "__main__":
    main()
