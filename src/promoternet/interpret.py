"""Interpretability tools: gradient saliency, conv-1 filter motif extraction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from promoternet.features import one_hot_batch


def gradient_saliency(
    model: nn.Module,
    sequences: list[str],
    seq_length: int = 150,
    device: str | None = None,
) -> np.ndarray:
    """Input × gradient saliency.

    Returns (N, 4, L) array; per-position importance is typically computed as
    summing the absolute saliency across the 4 channels.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()

    X = one_hot_batch(sequences, length=seq_length)
    xb = torch.from_numpy(X).to(dev).requires_grad_(True)

    preds = model(xb)
    grads = torch.autograd.grad(preds.sum(), xb, create_graph=False)[0]
    saliency = (xb * grads).detach().cpu().numpy()
    return saliency


def aggregate_position_saliency(saliency: np.ndarray) -> np.ndarray:
    """Reduce (N, 4, L) saliency to (L,) by mean of per-sequence position |saliency|."""
    per_seq_per_pos = np.abs(saliency).sum(axis=1)
    return per_seq_per_pos.mean(axis=0)


def extract_conv1_filters_as_pwms(
    model: nn.Module,
    sequences: list[str],
    seq_length: int = 150,
    activation_quantile: float = 0.99,
    device: str | None = None,
) -> dict[int, np.ndarray]:
    """Convert convolutional filters to PWM-like motifs.

    For each filter f in model.conv1:
      - run all sequences through conv1 (no nonlinearity)
      - find positions where activation exceeds the activation_quantile
      - extract the underlying one-hot patch
      - average → PWM (4, kernel_size)

    Returns dict mapping filter index -> (4, kernel_size) ACGT-channel PWM.
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(dev).eval()

    X = one_hot_batch(sequences, length=seq_length)
    xb = torch.from_numpy(X).to(dev)

    with torch.no_grad():
        activations = model.conv1(xb).cpu().numpy()  # (N, F, L_out)

    n_filters = activations.shape[1]
    kernel_size = model.conv1.kernel_size[0]
    padding = model.conv1.padding[0]

    pwms: dict[int, np.ndarray] = {}
    X_np = X
    for f in range(n_filters):
        flat = activations[:, f, :].ravel()
        threshold = float(np.quantile(flat, activation_quantile))
        if threshold <= 0:
            continue

        accum = np.zeros((4, kernel_size), dtype=np.float64)
        n_hits = 0
        n_seq, _, L_out = activations.shape
        for n in range(n_seq):
            for j in range(L_out):
                if activations[n, f, j] < threshold:
                    continue
                start = j - padding
                end = start + kernel_size
                if start < 0 or end > seq_length:
                    continue
                accum += X_np[n, :, start:end]
                n_hits += 1
        if n_hits == 0:
            continue
        pwms[f] = accum / n_hits
    return pwms


def motif_match_score(pwm: np.ndarray, consensus: str) -> float:
    """Best-aligned cosine similarity between an extracted PWM and a consensus motif.

    Slides the consensus across the PWM kernel and reports the maximum cosine,
    so a 6-bp motif can match anywhere inside a 12-bp filter.
    """
    target = one_hot_batch([consensus], length=len(consensus))[0]  # (4, k)
    k = target.shape[1]
    if pwm.shape[1] < k:
        return 0.0
    target_norm = target / (np.linalg.norm(target) + 1e-12)
    best = 0.0
    for offset in range(pwm.shape[1] - k + 1):
        window = pwm[:, offset : offset + k]
        a = window / (np.linalg.norm(window) + 1e-12)
        score = float((a * target_norm).sum())
        if score > best:
            best = score
    return best


def find_motif_filters(
    pwms: dict[int, np.ndarray],
    consensus_motifs: dict[str, str],
    threshold: float = 0.50,
) -> dict[str, list[tuple[int, float]]]:
    """For each named consensus motif, list (filter_idx, score) pairs above threshold."""
    matches: dict[str, list[tuple[int, float]]] = {name: [] for name in consensus_motifs}
    for f, pwm in pwms.items():
        for name, consensus in consensus_motifs.items():
            score = motif_match_score(pwm, consensus)
            if score >= threshold:
                matches[name].append((f, score))
    for name in matches:
        matches[name].sort(key=lambda x: -x[1])
    return matches


def top_high_expression_sequences(df: pd.DataFrame, n: int = 200) -> list[str]:
    return df.nlargest(n, "log_expression")["sequence"].tolist()
