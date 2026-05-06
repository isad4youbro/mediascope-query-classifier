import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, fbeta_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solution import PredictionModel


CONTENT_CLASSES = ["", "фильм", "сериал", "мультфильм", "мультсериал", "прочее"]


def normalize_text(x: object) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip().lower()


def title_tokens(s: str) -> Counter:
    tokens = re.findall(r"[a-zа-я0-9]+", normalize_text(s))
    return Counter(tokens)


def token_f1_single(true_title: str, pred_title: str) -> float:
    true_counts = title_tokens(true_title)
    pred_counts = title_tokens(pred_title)

    if not true_counts and not pred_counts:
        return 1.0
    if not true_counts or not pred_counts:
        return 0.0

    common = sum((true_counts & pred_counts).values())
    pred_total = sum(pred_counts.values())
    true_total = sum(true_counts.values())

    if pred_total == 0 or true_total == 0:
        return 0.0

    precision = common / pred_total
    recall = common / true_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_metrics(df_true: pd.DataFrame, df_pred: pd.DataFrame) -> dict[str, float]:
    y_true_type = df_true["TypeQuery"].astype(int)
    y_pred_type = df_pred["TypeQuery"].astype(int)

    typequery_f2 = fbeta_score(y_true_type, y_pred_type, beta=2, zero_division=0)

    mask_video = y_true_type == 1

    true_ct = df_true.loc[mask_video, "ContentType"].map(normalize_text)
    pred_ct = df_pred.loc[mask_video, "ContentType"].map(normalize_text)

    contenttype_macro_f1 = f1_score(
        true_ct,
        pred_ct,
        labels=CONTENT_CLASSES,
        average="macro",
        zero_division=0,
    )

    true_title = df_true.loc[mask_video, "Title"].map(normalize_text)
    pred_title = df_pred.loc[mask_video, "Title"].map(normalize_text)

    title_scores = [token_f1_single(t, p) for t, p in zip(true_title, pred_title)]
    title_token_f1 = float(sum(title_scores) / len(title_scores)) if title_scores else 0.0

    combined = 0.35 * typequery_f2 + 0.30 * contenttype_macro_f1 + 0.35 * title_token_f1

    return {
        "typequery_f2": float(typequery_f2),
        "contenttype_macro_f1": float(contenttype_macro_f1),
        "title_token_f1": float(title_token_f1),
        "combined_score": float(combined),
    }


def run_model(model: PredictionModel, df_queries: pd.DataFrame) -> pd.DataFrame:
    chunks = []
    step = max(1, int(getattr(model, "batch_size", 10)))
    for start in range(0, len(df_queries), step):
        chunk = df_queries.iloc[start : start + step][["QueryText"]]
        chunks.append(model.predict(chunk))
    pred = pd.concat(chunks, ignore_index=True)
    pred = pred[["QueryText", "TypeQuery", "Title", "ContentType"]].copy()
    pred["TypeQuery"] = pred["TypeQuery"].fillna(0).astype(int).clip(0, 1)
    pred["Title"] = pred["Title"].map(normalize_text)
    pred["ContentType"] = pred["ContentType"].map(normalize_text)
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Local evaluator for mediascope solution.py")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "train.csv",
        help="Path to labeled CSV",
    )
    parser.add_argument("--head", type=int, default=0, help="Use first N rows (0=all)")
    args = parser.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")

    df = pd.read_csv(args.data)
    req = {"QueryText", "TypeQuery", "Title", "ContentType"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if args.head > 0:
        df = df.head(args.head).copy()

    model = PredictionModel()
    pred = run_model(model, df[["QueryText"]])

    true_df = df[["QueryText", "TypeQuery", "Title", "ContentType"]].copy()
    true_df["TypeQuery"] = true_df["TypeQuery"].astype(int)
    true_df["Title"] = true_df["Title"].map(normalize_text)
    true_df["ContentType"] = true_df["ContentType"].map(normalize_text)

    metrics = compute_metrics(true_df, pred)

    print("Rows:", len(df))
    print("typequery_f2        =", round(metrics["typequery_f2"], 6))
    print("contenttype_macro_f1=", round(metrics["contenttype_macro_f1"], 6))
    print("title_token_f1      =", round(metrics["title_token_f1"], 6))
    print("combined_score      =", round(metrics["combined_score"], 6))


if __name__ == "__main__":
    main()

