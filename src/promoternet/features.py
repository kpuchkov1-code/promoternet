"""Feature extraction utilities: one-hot encoding, k-mer counting, PWM scoring.

Sequences are assumed to be uppercase ACGT only. Validation is callers'
responsibility; the data loader filters out non-canonical bases upstream.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

NUCLEOTIDES: str = "ACGT"
NT_TO_IDX: dict[str, int] = {nt: i for i, nt in enumerate(NUCLEOTIDES)}


def one_hot_encode(seq: str, length: int | None = None) -> np.ndarray:
    """One-hot encode a single sequence as (4, L) float32.

    Channels are A, C, G, T in that order. Bases not in {A,C,G,T} are encoded
    as all-zero columns. If `length` is given and seq is shorter, right-pads
    with zeros; if longer, raises ValueError.
    """
    L = len(seq) if length is None else length
    if length is not None and len(seq) > length:
        raise ValueError(f"sequence length {len(seq)} > target length {length}")
    arr = np.zeros((4, L), dtype=np.float32)
    for i, nt in enumerate(seq):
        idx = NT_TO_IDX.get(nt)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr


def one_hot_batch(sequences: Iterable[str], length: int | None = None) -> np.ndarray:
    """Stack of one-hot encodings as (N, 4, L) float32."""
    seqs = list(sequences)
    if not seqs:
        return np.zeros((0, 4, 0), dtype=np.float32)
    L = length if length is not None else len(seqs[0])
    out = np.zeros((len(seqs), 4, L), dtype=np.float32)
    for n, seq in enumerate(seqs):
        if len(seq) > L:
            raise ValueError(f"sequence {n} has length {len(seq)} > {L}")
        for i, nt in enumerate(seq):
            idx = NT_TO_IDX.get(nt)
            if idx is not None:
                out[n, idx, i] = 1.0
    return out


def kmer_counts(seq: str, k: int) -> dict[str, int]:
    """Count overlapping k-mers in a single sequence."""
    counts: dict[str, int] = {}
    for i in range(len(seq) - k + 1):
        kmer = seq[i : i + k]
        counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def all_kmers(k: int) -> list[str]:
    """All canonical ACGT k-mers in lexicographic order. Length = 4**k."""
    if k == 0:
        return [""]
    smaller = all_kmers(k - 1)
    return [s + nt for s in smaller for nt in NUCLEOTIDES]


def kmer_feature_matrix(sequences: Iterable[str], k: int) -> np.ndarray:
    """(N, 4**k) matrix of k-mer counts. Rows are sequences, columns are k-mers."""
    vocab = all_kmers(k)
    vocab_idx = {kmer: i for i, kmer in enumerate(vocab)}
    seqs = list(sequences)
    out = np.zeros((len(seqs), len(vocab)), dtype=np.float32)
    for n, seq in enumerate(seqs):
        for i in range(len(seq) - k + 1):
            j = vocab_idx.get(seq[i : i + k])
            if j is not None:
                out[n, j] += 1.0
    return out


def kmer_feature_matrix_multi(sequences: Iterable[str], ks: Iterable[int]) -> np.ndarray:
    """Concatenate k-mer feature matrices for multiple k values along columns."""
    seqs = list(sequences)
    blocks = [kmer_feature_matrix(seqs, k) for k in ks]
    return np.concatenate(blocks, axis=1)


SIGMA70_MINUS_10_PWM: np.ndarray = np.array(
    [
        [0.05, 0.85, 0.05, 0.05],  # T (TATAAT[0])
        [0.85, 0.05, 0.05, 0.05],  # A
        [0.05, 0.05, 0.05, 0.85],  # T  -> stored as ACGT, T idx 3
        [0.85, 0.05, 0.05, 0.05],  # A
        [0.85, 0.05, 0.05, 0.05],  # A
        [0.05, 0.05, 0.05, 0.85],  # T
    ],
    dtype=np.float32,
).T  # shape (4, 6)


SIGMA70_MINUS_35_PWM: np.ndarray = np.array(
    [
        [0.05, 0.05, 0.05, 0.85],  # T (TTGACA)
        [0.05, 0.05, 0.05, 0.85],  # T
        [0.05, 0.05, 0.85, 0.05],  # G
        [0.85, 0.05, 0.05, 0.05],  # A
        [0.05, 0.85, 0.05, 0.05],  # C
        [0.85, 0.05, 0.05, 0.05],  # A
    ],
    dtype=np.float32,
).T  # shape (4, 6)


def _log_pwm(pwm: np.ndarray, pseudocount: float = 1e-3) -> np.ndarray:
    background = 0.25
    safe = pwm + pseudocount
    safe = safe / safe.sum(axis=0, keepdims=True)
    return np.log2(safe / background)


def best_pwm_score(seq: str, pwm: np.ndarray) -> tuple[float, int]:
    """Slide PWM across seq, return (max_log_likelihood_score, best_position)."""
    log_pwm = _log_pwm(pwm)
    L_seq, L_pwm = len(seq), pwm.shape[1]
    if L_seq < L_pwm:
        raise ValueError(f"sequence length {L_seq} < PWM length {L_pwm}")
    best_score = -np.inf
    best_pos = -1
    for start in range(L_seq - L_pwm + 1):
        score = 0.0
        for j in range(L_pwm):
            idx = NT_TO_IDX.get(seq[start + j])
            if idx is None:
                score = -np.inf
                break
            score += float(log_pwm[idx, j])
        if score > best_score:
            best_score = score
            best_pos = start
    return best_score, best_pos


def pwm_features(sequences: Iterable[str]) -> np.ndarray:
    """Return (N, 4) feature matrix: -10 score, -10 pos, -35 score, -35 pos.

    Useful as a tiny mechanistic baseline that captures whether the canonical
    σ70 boxes are present and well-positioned.
    """
    seqs = list(sequences)
    out = np.zeros((len(seqs), 4), dtype=np.float32)
    for n, seq in enumerate(seqs):
        s10, p10 = best_pwm_score(seq, SIGMA70_MINUS_10_PWM)
        s35, p35 = best_pwm_score(seq, SIGMA70_MINUS_35_PWM)
        out[n] = [s10, float(p10), s35, float(p35)]
    return out
