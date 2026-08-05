from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import regularizers


def build_minisea(
    *,
    sequence_length: int = 1000,
    l1: float = 1e-8,
    l2: float = 5e-7,
) -> tf.keras.Model:
    """Build the DeepSEA-inspired MiniSEA binary classifier."""
    elastic = regularizers.l1_l2(l1=l1, l2=l2)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(sequence_length, 4)),
            tf.keras.layers.Conv1D(
                filters=320,
                kernel_size=8,
                strides=1,
                activation="relu",
            ),
            tf.keras.layers.MaxPooling1D(pool_size=4, strides=4),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Conv1D(
                filters=480,
                kernel_size=8,
                strides=1,
                activation="relu",
            ),
            tf.keras.layers.MaxPooling1D(pool_size=4, strides=4),
            tf.keras.layers.Dropout(0.20),
            tf.keras.layers.Conv1D(
                filters=960,
                kernel_size=8,
                strides=1,
                activation="relu",
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(
                units=925,
                activation="relu",
                kernel_regularizer=elastic,
            ),
            tf.keras.layers.Dropout(0.50),
            tf.keras.layers.Dense(units=1, activation="sigmoid"),
        ],
        name="minisea",
    )

    return model


def compile_model(
    model: tf.keras.Model,
    *,
    learning_rate: float = 1e-3,
) -> None:
    """Compile MiniSEA with binary classification metrics."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auroc", curve="ROC"),
            tf.keras.metrics.AUC(name="auprc", curve="PR"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
