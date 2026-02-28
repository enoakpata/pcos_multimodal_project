import os
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import xgboost as xgb

from .utils import seed_everything, ensure_dir, mean_std


@dataclass
class ClinicalConfig:
    seed: int = 42
    n_splits: int = 5
    learning_rate: float = 0.05
    max_depth: int = 4
    n_estimators: int = 300
    subsample: float = 0.8
    colsample_bytree: float = 0.8


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "auc": roc_auc_score(y_true, y_prob),
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "specificity": tn / (tn + fp)
    }


def run_kfold_xgboost(df: pd.DataFrame, out_dir: str, cfg: ClinicalConfig):

    ensure_dir(out_dir)
    seed_everything(cfg.seed)

    X = df.drop(columns=["PCOS (Y/N)"])
    y = df["PCOS (Y/N)"].values

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):

        print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = xgb.XGBClassifier(
            objective="binary:logistic",
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            n_estimators=cfg.n_estimators,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=cfg.seed,
            eval_metric="logloss"
        )

        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_val)[:, 1]

        metrics = compute_metrics(y_val, y_prob)

        print(
            f"AUC: {metrics['auc']:.4f} | "
            f"ACC: {metrics['acc']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        fold_results.append(metrics)

        model.save_model(os.path.join(out_dir, f"xgb_fold{fold}.json"))

    results_df = pd.DataFrame(fold_results)
    results_df.to_csv(os.path.join(out_dir, "clinical_metrics_folds.csv"), index=False)

    auc_mean, auc_std = mean_std(results_df["auc"].tolist())
    acc_mean, acc_std = mean_std(results_df["acc"].tolist())
    f1_mean, f1_std = mean_std(results_df["f1"].tolist())

    print("\n========== 5-Fold Summary ==========")
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"F1 : {f1_mean:.4f} ± {f1_std:.4f}")

    return results_df
