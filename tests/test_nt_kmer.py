"""Test the 6-mer tokenization helper used for DNABERT-6 inputs."""

from __future__ import annotations

from promoternet.nt_model import KMER_LEN, _seq_to_kmers


def test_seq_to_kmers_overlapping():
    seq = "ACGTACGTAC"  # 10 bp
    out = _seq_to_kmers(seq, k=6)
    expected = "ACGTAC CGTACG GTACGT TACGTA ACGTAC"
    assert out == expected
    assert len(out.split()) == len(seq) - KMER_LEN + 1


def test_seq_to_kmers_default_kmer_len():
    seq = "A" * 10
    out = _seq_to_kmers(seq)
    assert len(out.split()) == 5


def test_seq_to_kmers_short_sequence_returns_empty():
    assert _seq_to_kmers("ACGT", k=6) == ""
