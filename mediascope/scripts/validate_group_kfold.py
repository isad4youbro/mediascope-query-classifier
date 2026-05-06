import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution import PredictionModel
from local_eval import compute_metrics, normalize_text, run_model


def norm_group_key(text: str) -> str:
    text = str(text).lower().replace("ё", "е")
    text = " ".join(text.split())
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Leak-safe group k-fold validation")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "train.csv",
        help="Path to labeled CSV",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of GroupKFold splits")
    parser.add_argument("--max_rows", type=int, default=0, help="Optional cap on number of rows")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    df = pd.read_csv(args.data)
    req = {"QueryText", "TypeQuery", "Title", "ContentType"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    groups = df["QueryText"].map(norm_group_key)
    gkf = GroupKFold(n_splits=args.folds)

    metrics_by_fold = []
    original_train_cache_path = os.environ.get("TRAIN_CACHE_PATH")
    temp_root = ROOT / ".validation_tmp"
    temp_root.mkdir(parents=True, exist_ok=True)

    try:
        for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, groups=groups), start=1):
            train_df = df.iloc[tr_idx].copy()
            val_df = df.iloc[va_idx].copy()

            fold_dir = temp_root / f"fold_{fold}"
            if fold_dir.exists():
                shutil.rmtree(fold_dir, ignore_errors=True)
            fold_dir.mkdir(parents=True, exist_ok=True)

            train_path = fold_dir / "train_fold.csv"
            train_df[["QueryText", "TypeQuery", "Title", "ContentType"]].to_csv(train_path, index=False)

            os.environ["TRAIN_CACHE_PATH"] = str(train_path)
            model = PredictionModel()
            pred = run_model(model, val_df[["QueryText"]]).reset_index(drop=True)

            true_df = val_df[["QueryText", "TypeQuery", "Title", "ContentType"]].copy().reset_index(drop=True)
            true_df["TypeQuery"] = true_df["TypeQuery"].astype(int)
            true_df["Title"] = true_df["Title"].map(normalize_text)
            true_df["ContentType"] = true_df["ContentType"].map(normalize_text)

            m = compute_metrics(true_df, pred)
            metrics_by_fold.append(m)

            print(
                f"Fold {fold}/{args.folds}: "
                f"combined={m['combined_score']:.6f}, "
                f"type_f2={m['typequery_f2']:.6f}, "
                f"ct_f1={m['contenttype_macro_f1']:.6f}, "
                f"title_f1={m['title_token_f1']:.6f}, "
                f"rows={len(val_df)}"
            )
    finally:
        if original_train_cache_path is None:
            os.environ.pop("TRAIN_CACHE_PATH", None)
        else:
            os.environ["TRAIN_CACHE_PATH"] = original_train_cache_path
        shutil.rmtree(temp_root, ignore_errors=True)

    if not metrics_by_fold:
        print("No folds were evaluated.")
        return

    keys = ["combined_score", "typequery_f2", "contenttype_macro_f1", "title_token_f1"]
    print("\nMean ± std across folds:")
    for k in keys:
        vals = np.array([x[k] for x in metrics_by_fold], dtype=float)
        print(f"{k:22s} = {vals.mean():.6f} ± {vals.std(ddof=0):.6f}")


if __name__ == "__main__":
    main()

