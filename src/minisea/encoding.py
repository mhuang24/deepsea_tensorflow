from __future__ import annotations

import numpy as np

BASE_TO_VECTOR = {
    "A": (1.0, 0.0, 0.0, 0.0),
    "C": (0.0, 1.0, 0.0, 0.0),
    "G": (0.0, 0.0, 1.0, 0.0),
    "T": (0.0, 0.0, 0.0, 1.0),
}


def encode_sequence(sequence: str, *, allow_unknown: bool = False) -> np.ndarray:
    """One-hot encode a DNA sequence as an array with shape (length, 4).

    Bases are ordered as A, C, G, T.

    Args:
        sequence: DNA sequence to encode.
        allow_unknown: When True, unknown bases are encoded as all zeros.
            When False, unknown bases raise ValueError.

    Returns:
        A float32 NumPy array with shape (len(sequence), 4).
    """
    sequence = sequence.upper()
    encoded = np.zeros((len(sequence), 4), dtype=np.float32)

    for index, base in enumerate(sequence):
        vector = BASE_TO_VECTOR.get(base)
        if vector is None:
            if allow_unknown:
                continue
            raise ValueError(
                f"Unsupported DNA base {base!r} at position {index}. "
                "Expected only A, C, G, or T."
            )
        encoded[index] = vector

    return encoded


def encode_sequences(sequences: list[str]) -> np.ndarray:
    """Encode equal-length DNA sequences into shape (samples, length, 4)."""
    if not sequences:
        raise ValueError("At least one sequence is required.")

    sequence_length = len(sequences[0])
    if any(len(sequence) != sequence_length for sequence in sequences):
        raise ValueError("All sequences must have the same length.")

    return np.stack([encode_sequence(sequence) for sequence in sequences])
