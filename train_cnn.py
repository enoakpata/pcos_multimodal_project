import os
import argparse
import pandas as pd

from src.cnn_model import run_kfold_cnn, CNNConfig


def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNet-B0 with 5-fold CV (PCOS ultrasound)")
    p.add_argument("--project_dir", type=str, default=".", help="Root project directory")
    p.add_argument("--train_csv", type=str, default="data/images/clean_train_split.csv")
    p.add_argument("--test_csv", type=str, default="data/images/clean_test_split.csv")
    p.add_argument("--out_dir", type=str, default="outputs/cnn")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_pretrained", action="store_true", help="Train without ImageNet weights")
    return p.parse_args()


def main():
    args = parse_args()

    project_dir = os.path.abspath(args.project_dir)
    train_csv = os.path.join(project_dir, args.train_csv)
    test_csv  = os.path.join(project_dir, args.test_csv)
    out_dir   = os.path.join(project_dir, args.out_dir)

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)
    df_images = pd.concat([train_df, test_df]).reset_index(drop=True)

    cfg = CNNConfig(
        seed=args.seed,
        n_splits=args.folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        use_pretrained=(not args.no_pretrained),
    )

    run_kfold_cnn(df_images=df_images, out_dir=out_dir, cfg=cfg)


if __name__ == "__main__":
    main()
