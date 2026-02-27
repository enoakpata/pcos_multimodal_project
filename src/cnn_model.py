import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from sklearn.model_selection import StratifiedKFold

from .dataset import PCOSDataset
from .utils import seed_everything, compute_binary_metrics, ensure_dir, mean_std


@dataclass
class CNNConfig:
    seed: int = 42
    n_splits: int = 5
    epochs: int = 5
    batch_size: int = 32
    lr: float = 1e-4
    num_workers: int = 2
    threshold: float = 0.5
    use_pretrained: bool = True  # reviewers accept this
    image_size: int = 224


def build_transforms(image_size: int = 224) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.GaussianBlur(3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tfms, val_tfms


def build_model(use_pretrained: bool = True) -> nn.Module:
    if use_pretrained:
        weights = EfficientNet_B0_Weights.DEFAULT
        model = efficientnet_b0(weights=weights)
    else:
        model = efficientnet_b0(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)  # binary logit
    return model


def _train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    model.train()
    all_probs, all_true = [], []
    running_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1, 1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        all_probs.append(probs)
        all_true.append(y.detach().cpu().numpy().ravel())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)

    metrics = compute_binary_metrics(y_true, y_prob, threshold=0.5)
    metrics["loss"] = float(running_loss / max(n_batches, 1))
    return metrics


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float
) -> Dict[str, float]:
    model.eval()
    all_probs, all_true = [], []
    running_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).view(-1, 1)

        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        all_probs.append(probs)
        all_true.append(y.detach().cpu().numpy().ravel())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_true)

    metrics = compute_binary_metrics(y_true, y_prob, threshold=threshold)
    metrics["loss"] = float(running_loss / max(n_batches, 1))
    return metrics


def run_kfold_cnn(
    df_images: pd.DataFrame,
    out_dir: str,
    cfg: CNNConfig
) -> pd.DataFrame:
    """
    df_images must include: path, label
    Saves:
      - fold checkpoints in out_dir/checkpoints/
      - metrics CSV in out_dir/metrics_folds.csv
    Returns a DataFrame of fold results.
    """
    ensure_dir(out_dir)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    ensure_dir(ckpt_dir)

    seed_everything(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    train_tfms, val_tfms = build_transforms(cfg.image_size)

    X = df_images["path"].values
    y = df_images["label"].values.astype(int)

    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold_rows: List[Dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        fold_start = time.time()
        print(f"\n========== Fold {fold}/{cfg.n_splits} ==========")

        train_df = df_images.iloc[tr_idx].reset_index(drop=True)
        val_df   = df_images.iloc[va_idx].reset_index(drop=True)

        # pos_weight computed from TRAIN fold only
        n_neg = int((train_df["label"] == 0).sum())
        n_pos = int((train_df["label"] == 1).sum())
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

        train_ds = PCOSDataset(train_df, transform=train_tfms, strict=True)
        val_ds   = PCOSDataset(val_df, transform=val_tfms, strict=True)

        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=pin_memory
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.batch_size, shuffle=False,
            num_workers=cfg.num_workers, pin_memory=pin_memory
        )

        model = build_model(cfg.use_pretrained).to(device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

        best_auc = -1.0
        best_state = None

        for epoch in range(1, cfg.epochs + 1):
            tr_metrics = _train_one_epoch(model, train_loader, optimizer, criterion, device)
            va_metrics = _evaluate(model, val_loader, criterion, device, threshold=cfg.threshold)

            print(
                f"Epoch {epoch:02d}/{cfg.epochs} | "
                f"Train loss {tr_metrics['loss']:.4f} AUC {tr_metrics['auc']:.4f} ACC {tr_metrics['acc']:.4f} | "
                f"Val   loss {va_metrics['loss']:.4f} AUC {va_metrics['auc']:.4f} ACC {va_metrics['acc']:.4f}"
            )

            if not np.isnan(va_metrics["auc"]) and va_metrics["auc"] > best_auc:
                best_auc = va_metrics["auc"]
                best_state = {
                    "model_state": model.state_dict(),
                    "fold": fold,
                    "epoch": epoch,
                    "best_val_auc": best_auc,
                    "pos_weight": float(pos_weight.item()),
                    "config": cfg.__dict__,
                }

        # Save best checkpoint for fold
        ckpt_path = os.path.join(ckpt_dir, f"efficientnetb0_fold{fold}.pt")
        torch.save(best_state, ckpt_path)

        # Evaluate best state again for reporting consistency
        model.load_state_dict(best_state["model_state"])
        final_val = _evaluate(model, val_loader, criterion, device, threshold=cfg.threshold)

        row = {
            "fold": float(fold),
            "val_auc": float(final_val["auc"]),
            "val_acc": float(final_val["acc"]),
            "val_precision": float(final_val["precision"]),
            "val_recall": float(final_val["recall"]),
            "val_f1": float(final_val["f1"]),
            "val_specificity": float(final_val["specificity"]),
            "val_loss": float(final_val["loss"]),
            "pos_weight": float(pos_weight.item()),
            "seconds": float(time.time() - fold_start),
        }
        fold_rows.append(row)

        print(f"Fold {fold} BEST Val AUC: {row['val_auc']:.4f} | ACC: {row['val_acc']:.4f}")
        print(f"Checkpoint: {ckpt_path}")

    results = pd.DataFrame(fold_rows)
    results_path = os.path.join(out_dir, "metrics_folds.csv")
    results.to_csv(results_path, index=False)

    # Print summary
    auc_mean, auc_std = mean_std(results["val_auc"].tolist())
    acc_mean, acc_std = mean_std(results["val_acc"].tolist())
    f1_mean, f1_std = mean_std(results["val_f1"].tolist())

    print("\n========== 5-Fold Summary ==========")
    print(f"AUC: {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"ACC: {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"F1 : {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"Saved fold metrics: {results_path}")

    return results
