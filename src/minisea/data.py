from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import pandas as pd
from intervaltree import IntervalTree
from pyfaidx import Fasta

from minisea.encoding import encode_sequences


@dataclass(frozen=True)
class DatasetSplit:
    x: np.ndarray
    y: np.ndarray
    chromosomes: np.ndarray


def normalize_chromosome(chromosome: str) -> str:
    chromosome = str(chromosome).strip()
    return chromosome if chromosome.startswith("chr") else f"chr{chromosome}"


def load_peak_intervals(
    bed_path: str | Path,
    *,
    max_peaks: int | None = None,
) -> dict[str, IntervalTree]:
    """Load BED intervals into one IntervalTree per chromosome."""
    bed_path = Path(bed_path)
    if not bed_path.exists():
        raise FileNotFoundError(f"BED file not found: {bed_path}")

    bed = pd.read_csv(
        bed_path,
        sep="\t",
        header=None,
        comment="#",
        usecols=[0, 1, 2],
        names=["chromosome", "start", "end"],
    )

    if max_peaks is not None:
        bed = bed.head(max_peaks)

    trees: dict[str, IntervalTree] = defaultdict(IntervalTree)

    for row in bed.itertuples(index=False):
        chromosome = normalize_chromosome(row.chromosome)
        start = int(row.start)
        end = int(row.end)

        if start < 0 or end <= start:
            continue

        trees[chromosome].addi(start, end)

    return dict(trees)


def extract_centered_positive_sequences(
    genome: Fasta,
    intervals: dict[str, IntervalTree],
    *,
    sequence_length: int = 1000,
) -> list[tuple[str, str]]:
    """Extract fixed-length sequences centered on positive BED intervals."""
    if sequence_length <= 0 or sequence_length % 2 != 0:
        raise ValueError("sequence_length must be a positive even integer.")

    half_length = sequence_length // 2
    examples: list[tuple[str, str]] = []

    for chromosome, tree in intervals.items():
        if chromosome not in genome:
            continue

        chromosome_length = len(genome[chromosome])

        for interval in sorted(tree):
            midpoint = (interval.begin + interval.end) // 2
            start = midpoint - half_length
            end = midpoint + half_length

            if start < 0 or end > chromosome_length:
                continue

            sequence = str(genome[chromosome][start:end]).upper()

            if len(sequence) != sequence_length or "N" in sequence:
                continue

            examples.append((chromosome, sequence))

    return examples


def sample_negative_sequences(
    genome: Fasta,
    positive_intervals: dict[str, IntervalTree],
    target_counts: dict[str, int],
    *,
    sequence_length: int = 1000,
    seed: int = 42,
    max_attempt_multiplier: int = 100,
) -> list[tuple[str, str]]:
    """Sample non-overlapping negative sequences by chromosome."""
    rng = random.Random(seed)
    negatives: list[tuple[str, str]] = []

    for chromosome, target_count in target_counts.items():
        if target_count <= 0 or chromosome not in genome:
            continue

        chromosome_length = len(genome[chromosome])
        if chromosome_length <= sequence_length:
            continue

        tree = positive_intervals.get(chromosome, IntervalTree())
        sampled = 0
        attempts = 0
        max_attempts = max(target_count * max_attempt_multiplier, 1000)

        while sampled < target_count and attempts < max_attempts:
            attempts += 1
            start = rng.randint(0, chromosome_length - sequence_length)
            end = start + sequence_length

            if tree.overlap(start, end):
                continue

            sequence = str(genome[chromosome][start:end]).upper()

            if len(sequence) != sequence_length or "N" in sequence:
                continue

            negatives.append((chromosome, sequence))
            sampled += 1

        if sampled < target_count:
            raise RuntimeError(
                f"Could only sample {sampled}/{target_count} negative examples "
                f"for {chromosome} after {attempts} attempts."
            )

    return negatives


def _counts_by_chromosome(
    examples: Iterable[tuple[str, str]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for chromosome, _ in examples:
        counts[chromosome] += 1
    return dict(counts)


def make_split(
    positive_examples: list[tuple[str, str]],
    negative_examples: list[tuple[str, str]],
    chromosomes: set[str],
    *,
    seed: int,
) -> DatasetSplit:
    """Create and shuffle one labeled dataset split."""
    selected_positive = [
        (chromosome, sequence)
        for chromosome, sequence in positive_examples
        if chromosome in chromosomes
    ]
    selected_negative = [
        (chromosome, sequence)
        for chromosome, sequence in negative_examples
        if chromosome in chromosomes
    ]

    examples = selected_positive + selected_negative
    if not examples:
        raise ValueError(
            f"No examples were found for chromosomes: {sorted(chromosomes)}"
        )

    sequences = [sequence for _, sequence in examples]
    chromosome_values = np.array(
        [chromosome for chromosome, _ in examples],
        dtype=str,
    )
    labels = np.concatenate(
        [
            np.ones(len(selected_positive), dtype=np.float32),
            np.zeros(len(selected_negative), dtype=np.float32),
        ]
    )

    encoded = encode_sequences(sequences)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(labels))

    return DatasetSplit(
        x=encoded[order],
        y=labels[order],
        chromosomes=chromosome_values[order],
    )


def build_chromosome_splits(
    genome_path: str | Path,
    bed_path: str | Path,
    *,
    validation_chromosome: str = "chr7",
    test_chromosome: str = "chr8",
    sequence_length: int = 1000,
    max_peaks: int | None = None,
    seed: int = 42,
) -> tuple[DatasetSplit, DatasetSplit, DatasetSplit]:
    """Build train, validation, and test splits held out by chromosome."""
    validation_chromosome = normalize_chromosome(validation_chromosome)
    test_chromosome = normalize_chromosome(test_chromosome)

    if validation_chromosome == test_chromosome:
        raise ValueError("Validation and test chromosomes must be different.")

    genome_path = Path(genome_path)
    if not genome_path.exists():
        raise FileNotFoundError(f"Genome FASTA not found: {genome_path}")

    genome = Fasta(str(genome_path), as_raw=True, sequence_always_upper=True)
    positive_intervals = load_peak_intervals(bed_path, max_peaks=max_peaks)

    positive_examples = extract_centered_positive_sequences(
        genome,
        positive_intervals,
        sequence_length=sequence_length,
    )

    if not positive_examples:
        raise RuntimeError("No valid positive sequences were extracted.")

    negative_examples = sample_negative_sequences(
        genome,
        positive_intervals,
        _counts_by_chromosome(positive_examples),
        sequence_length=sequence_length,
        seed=seed,
    )

    available_chromosomes = {
        chromosome for chromosome, _ in positive_examples
    }

    held_out = {validation_chromosome, test_chromosome}
    train_chromosomes = available_chromosomes - held_out

    if not train_chromosomes:
        raise RuntimeError("No chromosomes remain for training.")

    train = make_split(
        positive_examples,
        negative_examples,
        train_chromosomes,
        seed=seed,
    )
    validation = make_split(
        positive_examples,
        negative_examples,
        {validation_chromosome},
        seed=seed + 1,
    )
    test = make_split(
        positive_examples,
        negative_examples,
        {test_chromosome},
        seed=seed + 2,
    )

    return train, validation, test


def save_split(path: str | Path, split: DatasetSplit) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=split.x,
        y=split.y,
        chromosomes=split.chromosomes,
    )


def load_split(path: str | Path) -> DatasetSplit:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    data = np.load(path)
    return DatasetSplit(
        x=data["x"],
        y=data["y"],
        chromosomes=data["chromosomes"],
    )
