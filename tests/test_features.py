"""Tests for feature extraction."""

from __future__ import annotations

import numpy as np

from promoternet.features import (
    SIGMA70_MINUS_10_PWM,
    all_kmers,
    best_pwm_score,
    kmer_counts,
    kmer_feature_matrix,
    one_hot_batch,
    one_hot_encode,
    pwm_features,
)


def test_one_hot_encode_acgt():
    arr = one_hot_encode("ACGT")
    expected = np.eye(4, dtype=np.float32)
    np.testing.assert_array_equal(arr, expected)


def test_one_hot_encode_skips_n():
    arr = one_hot_encode("ANCGT")
    assert arr.shape == (4, 5)
    assert arr[:, 1].sum() == 0.0  # column for 'N' is all zero
    assert arr[0, 0] == 1.0


def test_one_hot_encode_pad_to_length():
    arr = one_hot_encode("AC", length=5)
    assert arr.shape == (4, 5)
    assert arr[0, 0] == 1.0
    assert arr[1, 1] == 1.0
    np.testing.assert_array_equal(arr[:, 2:], np.zeros((4, 3), dtype=np.float32))


def test_one_hot_batch_shape():
    arr = one_hot_batch(["ACGT", "TGCA"])
    assert arr.shape == (2, 4, 4)
    assert arr[0, 0, 0] == 1.0
    assert arr[1, 3, 0] == 1.0


def test_kmer_counts_overlapping():
    counts = kmer_counts("AAAA", k=2)
    assert counts == {"AA": 3}


def test_kmer_counts_mixed():
    counts = kmer_counts("ACGTACGT", k=3)
    assert counts["ACG"] == 2
    assert counts["CGT"] == 2
    assert counts["GTA"] == 1
    assert counts["TAC"] == 1


def test_all_kmers_length():
    assert len(all_kmers(1)) == 4
    assert len(all_kmers(3)) == 64
    assert len(all_kmers(4)) == 256


def test_kmer_feature_matrix_shape():
    seqs = ["ACGTACGT", "AAAAAAAA"]
    mat = kmer_feature_matrix(seqs, k=3)
    assert mat.shape == (2, 64)
    aaa_idx = all_kmers(3).index("AAA")
    assert mat[1, aaa_idx] == 6  # 8 - 3 + 1 = 6 occurrences of AAA in AAAAAAAA


def test_pwm_recovers_canonical_minus10_box():
    seq = "GGGGGGGGGGTATAATGGGGGG"  # consensus -10 at offset 10
    score, pos = best_pwm_score(seq, SIGMA70_MINUS_10_PWM)
    assert pos == 10
    assert score > 0  # consensus should beat background


def test_pwm_features_returns_4_columns():
    out = pwm_features(["GGGGGGGGGGTATAATGGGGGGGGGGGGGGTTGACAGGGGGGGGG"])
    assert out.shape == (1, 4)
    assert out[0, 0] > 0  # -10 box present
    assert out[0, 2] > 0  # -35 box present
