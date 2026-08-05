from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from pyfaidx import Fasta

from minisea.data import normalize_chromosome
from minisea.encoding import encode_sequence


def extract_centered_sequence(
    genome: Fasta,
    chromosome: str,
    position: int,
    *,
    sequence_length: int = 1000,
) -> str:
    chromosome = normalize_chromosome(chromosome)

    if chromosome not in genome:
        raise ValueError(f"Chromosome {chromosome!r} is not in the FASTA.")

    if sequence_length <= 0 or sequence_length % 2 != 0:
        raise ValueError("sequence_length must be a positive even integer.")

    half_length = sequence_length // 2
    start = position - half_length
    end = position + half_length
    chromosome_length = len(genome[chromosome])

    if start < 0 or end > chromosome_length:
        raise ValueError(
            f"Position {position} does not permit a {sequence_length}-bp "
            f"window on {chromosome} of length {chromosome_length}."
        )

    sequence = str(genome[chromosome][start:end]).upper()

    if len(sequence) != sequence_length:
        raise RuntimeError("Extracted sequence has an unexpected length.")

    if "N" in sequence:
        raise ValueError("Extracted sequence contains unknown bases (N).")

    return sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict DNase accessibility for one genomic coordinate."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--genome", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--position", required=True, type=int)
    parser.add_argument("--sequence-length", type=int, default=1000)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    genome_path = Path(args.genome)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not genome_path.exists():
        raise FileNotFoundError(f"Genome FASTA not found: {genome_path}")

    genome = Fasta(
        str(genome_path),
        as_raw=True,
        sequence_always_upper=True,
    )

    sequence = extract_centered_sequence(
        genome,
        args.chrom,
        args.position,
        sequence_length=args.sequence_length,
    )
    encoded = np.expand_dims(encode_sequence(sequence), axis=0)

    model = tf.keras.models.load_model(model_path)
    probability = float(model.predict(encoded, verbose=0).reshape(-1)[0])
    prediction = int(probability >= args.threshold)

    result = {
        "chromosome": normalize_chromosome(args.chrom),
        "position": args.position,
        "sequence_length": args.sequence_length,
        "probability": probability,
        "threshold": args.threshold,
        "prediction": "positive" if prediction else "negative",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
