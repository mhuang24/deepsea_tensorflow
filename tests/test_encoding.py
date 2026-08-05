import numpy as np
import pytest

from minisea.encoding import encode_sequence, encode_sequences


def test_encode_sequence() -> None:
    encoded = encode_sequence("ACGT")

    expected = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(encoded, expected)


def test_encode_sequence_is_case_insensitive() -> None:
    np.testing.assert_array_equal(
        encode_sequence("acgt"),
        encode_sequence("ACGT"),
    )


def test_encode_sequence_rejects_unknown_base() -> None:
    with pytest.raises(ValueError):
        encode_sequence("ACNT")


def test_encode_sequences_rejects_mixed_lengths() -> None:
    with pytest.raises(ValueError):
        encode_sequences(["ACGT", "ACG"])
