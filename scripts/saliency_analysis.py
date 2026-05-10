"""Saliency + filter-motif analysis on the trained CNN.

Loads the random-split CNN checkpoint, computes gradient saliency on
high-expression promoters, plots mean position saliency, and tests whether
any conv-1 filter rediscovers the canonical sigma70 -10 (TATAAT) or -35
(TTGACA) consensus motifs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from promoternet.data import load_cached, random_split
from promoternet.interpret import (
    aggregate_position_saliency,
    extract_conv1_filters_as_pwms,
    find_motif_filters,
    gradient_saliency,
    top_high_expression_sequences,
)
from promoternet.models import PromoterCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LENGTH = 150
RANDOM_SEED = 42
SIGMA70_CONSENSUS = {"-10 box (TATAAT)": "TATAAT", "-35 box (TTGACA)": "TTGACA"}


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    df = load_cached("urtecho2019")
    splits = random_split(df, val_frac=0.1, test_frac=0.1, seed=RANDOM_SEED)
    print(f"loaded {len(df):,d} rows  (test={len(splits['test']):,d})")

    model = PromoterCNN(seq_len=SEQ_LENGTH)
    ckpt = CHECKPOINTS_DIR / "cnn_random_8_1_1.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model = model.to(device).eval()
    print(f"loaded checkpoint: {ckpt.name}")

    high_expr = top_high_expression_sequences(splits["test"], n=200)
    print(f"computing saliency on top-{len(high_expr)} test promoters ...")
    saliency = gradient_saliency(model, high_expr, seq_length=SEQ_LENGTH, device=device)
    pos_importance = aggregate_position_saliency(saliency)

    fig, ax = plt.subplots(figsize=(11, 3.0))
    ax.plot(pos_importance, color="#3f3f3f", lw=1.0)
    ax.fill_between(np.arange(len(pos_importance)), 0, pos_importance, color="#3f3f3f", alpha=0.25)
    ax.set_xlabel("position in 150 bp promoter")
    ax.set_ylabel("mean |input × gradient|")
    ax.set_title("CNN saliency on top-200 high-expression test promoters")
    top_positions = np.argsort(pos_importance)[-5:][::-1]
    ymax = pos_importance.max()
    for pos in top_positions:
        ax.axvline(pos, color="crimson", lw=0.8, ls="--", alpha=0.6)
        ax.annotate(
            f"pos {pos}",
            xy=(pos, pos_importance[pos]),
            xytext=(pos, ymax * 1.05),
            ha="center",
            fontsize=8,
            color="crimson",
        )
    ax.set_ylim(0, ymax * 1.15)
    fig.tight_layout()
    saliency_fig = FIG_DIR / "saliency_position_importance.png"
    fig.savefig(saliency_fig, dpi=160)
    plt.close(fig)
    print(f"saved: {saliency_fig}")
    print(f"  top-5 attended positions: {sorted(int(p) for p in top_positions)}")

    print("\nextracting conv-1 filter motifs (this scans all training-style activations)...")
    train_seqs = splits["train"]["sequence"].sample(n=2000, random_state=0).tolist()
    pwms = extract_conv1_filters_as_pwms(
        model, train_seqs, seq_length=SEQ_LENGTH, activation_quantile=0.995, device=device
    )
    print(f"  recovered {len(pwms)} filter PWMs (kernel size = {model.conv1.kernel_size[0]})")

    matches = find_motif_filters(pwms, SIGMA70_CONSENSUS, threshold=0.45)
    motif_summary: dict[str, list[dict[str, float | int]]] = {}
    for name, hits in matches.items():
        print(f"\n{name}:")
        if not hits:
            print("  no filter above threshold 0.45")
        for f_idx, score in hits[:3]:
            pwm = pwms[f_idx]
            consensus = "".join("ACGT"[c] for c in pwm.argmax(axis=0))
            print(f"  filter {f_idx:3d}  cosine={score:.3f}  consensus={consensus}")
        motif_summary[name] = [{"filter": int(f), "score": round(s, 4)} for f, s in hits[:3]]

    summary_path = REPORTS_DIR / "saliency_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "top_attended_positions": sorted(int(p) for p in top_positions),
                "max_position_saliency": float(ymax),
                "n_filters_extracted": len(pwms),
                "motif_matches": motif_summary,
            },
            indent=2,
        )
    )
    print(f"\nsummary written to: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
