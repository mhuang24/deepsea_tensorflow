from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from minisea.data import build_chromosome_splits, save_split
from minisea.evaluate import save_evaluation_artifacts
from minisea.model import build_minisea, compile_model


def set_reproducibility(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_training_curves(
    history: tf.keras.callbacks.History,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    history_frame = pd.DataFrame(history.history)
    history_frame.to_csv(output_dir / "history.csv", index=False)

    if {"loss", "val_loss"}.issubset(history_frame.columns):
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(history_frame["loss"], label="train")
        axis.plot(history_frame["val_loss"], label="validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Loss")
        axis.set_title("Training and validation loss")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "loss_curve.png", dpi=200)
        plt.close(figure)

    if {"accuracy", "val_accuracy"}.issubset(history_frame.columns):
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(history_frame["accuracy"], label="train")
        axis.plot(history_frame["val_accuracy"], label="validation")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Accuracy")
        axis.set_title("Training and validation accuracy")
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / "accuracy_curve.png", dpi=200)
        plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MiniSEA DNase accessibility classifier."
    )
    parser.add_argument("--genome", required=True, help="Path to hg38 FASTA.")
    parser.add_argument("--peaks", required=True, help="Path to BED peaks.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-chromosome", default="chr7")
    parser.add_argument("--test-chromosome", default="chr8")
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument(
        "--max-peaks",
        type=int,
        default=None,
        help="Optional limit for faster development runs.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_reproducibility(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    print("Building chromosome-held-out datasets...")
    train, validation, test = build_chromosome_splits(
        args.genome,
        args.peaks,
        validation_chromosome=args.validation_chromosome,
        test_chromosome=args.test_chromosome,
        sequence_length=args.sequence_length,
        max_peaks=args.max_peaks,
        seed=args.seed,
    )

    save_split(output_dir / "train_data.npz", train)
    save_split(output_dir / "validation_data.npz", validation)
    save_split(output_dir / "test_data.npz", test)

    print(
        f"Train: {len(train.y):,} | "
        f"Validation: {len(validation.y):,} | "
        f"Test: {len(test.y):,}"
    )

    model = build_minisea(sequence_length=args.sequence_length)
    compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            min_delta=args.min_delta,
            mode="min",
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=output_dir / "model.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(output_dir / "training_log.csv"),
    ]

    history = model.fit(
        train.x,
        train.y,
        validation_data=(validation.x, validation.y),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(output_dir / "model.keras")
    save_training_curves(history, output_dir)

    probabilities = model.predict(test.x, verbose=0).reshape(-1)
    metrics = save_evaluation_artifacts(
        test,
        probabilities,
        output_dir,
    )

    print("Held-out test metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
