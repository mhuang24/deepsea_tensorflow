from intervaltree import IntervalTree
import numpy as np

from minisea.data import make_split, normalize_chromosome


def test_normalize_chromosome() -> None:
    assert normalize_chromosome("1") == "chr1"
    assert normalize_chromosome("chrX") == "chrX"


def test_make_split_filters_and_balances_examples() -> None:
    positives = [
        ("chr1", "A" * 10),
        ("chr2", "C" * 10),
    ]
    negatives = [
        ("chr1", "G" * 10),
        ("chr2", "T" * 10),
    ]

    split = make_split(
        positives,
        negatives,
        {"chr1"},
        seed=42,
    )

    assert split.x.shape == (2, 10, 4)
    assert sorted(split.y.tolist()) == [0.0, 1.0]
    assert set(split.chromosomes.tolist()) == {"chr1"}
