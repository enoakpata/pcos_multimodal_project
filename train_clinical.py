import os
import pandas as pd

from src.clinical_model import run_kfold_xgboost, ClinicalConfig

def main():
    project_dir = os.path.abspath(".")
    data_path = os.path.join(project_dir, "data/clinical/clinical_clean.csv")
    out_dir = os.path.join(project_dir, "outputs/clinical")

    df = pd.read_csv(data_path)

    cfg = ClinicalConfig()

    run_kfold_xgboost(df, out_dir, cfg)


if __name__ == "__main__":
    main()
