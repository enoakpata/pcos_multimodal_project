import os
import random
import numpy as np

from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import torch


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    y_true: shape (N,)
    y_prob: shape (N,) probabilities for positive class
    """
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    # Handle edge cases if a class is missing
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    else:
        tn = fp = fn = tp = 0
        specificity = float("nan")

    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")

    return {
        "auc": float(auc),
        "acc": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),  # sensitivity
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
    }


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.array(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))
