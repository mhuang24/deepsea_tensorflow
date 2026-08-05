from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

from minisea.data import DatasetSplit, load_split


def evaluate_predictions(
    labels: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities).reshape(-1)
    labels = np.asarray(labels).astype(int).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "precision": float(
            precision_score(labels, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(labels, predictions, zero_division=0)
        ),
        "threshold": float(threshold),
    }


def save_evaluation_artifacts(
    split: DatasetSplit,
    probabilities: np.ndarray,
    output_dir: str | Path,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = split.y.astype(int).reshape(-1)
    probabilities = np.asarray(probabilities).reshape(-1)
    predictions = (probabilities >= threshold).astype(int)

    metrics = evaluate_predictions(
        labels,
        probabilities,
        threshold=threshold,
    )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    figure, axis = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(labels, probabilities, ax=axis)
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(labels, probabilities, ax=axis)
    figure.tight_layout()
    figure.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=200,
    )
    plt.close(figure)

    matrix = confusion_matrix(labels, predictions)
    figure, axis = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(matrix).plot(ax=axis, values_format="d")
    figure.tight_layout()
    figure.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.close(figure)

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MiniSEA model."
    )
    parser.add_argument("--model", required=True, help="Path to .keras model.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a saved .npz dataset split.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for metrics and figures.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = tf.keras.models.load_model(args.model)
    split = load_split(args.dataset)
    probabilities = model.predict(split.x, verbose=0).reshape(-1)

    metrics = save_evaluation_artifacts(
        split,
        probabilities,
        args.output_dir,
        threshold=args.threshold,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
