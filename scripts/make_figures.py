"""Generate the figures used in the Neobe interview deck.

Outputs (PNG, 160 dpi):
  1. saliency_position_importance.png   (already produced by saliency_analysis.py)
  2. motif_logos.png                     CNN-rediscovered sigma70 -10 and -35 motifs
  3. calibration_random.png              CNN predictions vs measured (random split)
  4. calibration_lm10.png                CNN predictions on held-out -10 element
  5. generalization_bar.png              R2 by model x split bar chart
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import logomaker
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from promoternet.data import leave_element_out_split, load_cached, random_split
from promoternet.eval import regression_metrics
from promoternet.interpret import (
    extract_conv1_filters_as_pwms,
    find_motif_filters,
)
from promoternet.models import PromoterCNN
from promoternet.train import predict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LENGTH = 150
RANDOM_SEED = 42
SIGMA70_CONSENSUS = {"-10 box (TATAAT)": "TATAAT", "-35 box (TTGACA)": "TTGACA"}

NAVY = "#1f3a93"
RUST = "#c47828"
SLATE = "#3f3f3f"
CREAM = "#f9f5f0"


def _load_cnn(name: str, device: str) -> PromoterCNN:
    model = PromoterCNN(seq_len=SEQ_LENGTH)
    ckpt = CHECKPOINTS_DIR / f"cnn_{name}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return model.to(device).eval()


def make_motif_logos(device: str) -> None:
    print("\n[motif_logos.png]")
    df = load_cached("urtecho2019")
    splits = random_split(df, val_frac=0.1, test_frac=0.1, seed=RANDOM_SEED)
    model = _load_cnn("random_8_1_1", device)
    train_seqs = splits["train"]["sequence"].sample(n=2000, random_state=0).tolist()
    pwms = extract_conv1_filters_as_pwms(
        model, train_seqs, seq_length=SEQ_LENGTH, activation_quantile=0.995, device=device
    )
    matches = find_motif_filters(pwms, SIGMA70_CONSENSUS, threshold=0.5)

    fig, axes = plt.subplots(2, 2, figsize=(11, 5.0))
    fig.patch.set_facecolor(CREAM)
    for row, (name, hits) in enumerate(matches.items()):
        for col in range(2):
            ax = axes[row, col]
            ax.set_facecolor(CREAM)
            if col >= len(hits):
                ax.axis("off")
                continue
            f_idx, score = hits[col]
            pwm = pwms[f_idx]
            pwm_norm = pwm / (pwm.sum(axis=0, keepdims=True) + 1e-12)
            counts = (pwm_norm * 50).T
            df_logo = pd.DataFrame(counts, columns=list("ACGT"))
            logomaker.Logo(
                df_logo,
                ax=ax,
                color_scheme="classic",
                font_name="DejaVu Sans Mono",
            )
            ax.set_title(
                f"{name} — filter {f_idx}  (cosine = {score:.2f})",
                fontsize=10,
                color=SLATE,
            )
            ax.set_ylabel("information", fontsize=8, color=SLATE)
            ax.set_xticks([])
            for spine in ax.spines.values():
                spine.set_color(SLATE)
                spine.set_linewidth(0.5)
    fig.suptitle(
        "PromoterNet CNN rediscovers canonical σ70 -10 and -35 motifs without supervision",
        fontsize=12,
        color=NAVY,
        y=1.00,
    )
    fig.tight_layout()
    out = FIG_DIR / "motif_logos.png"
    fig.savefig(out, dpi=160, facecolor=CREAM)
    plt.close(fig)
    print(f"  saved: {out}")


def _scatter_calibration(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
) -> None:
    ax.scatter(
        y_true,
        y_pred,
        s=6,
        alpha=0.35,
        color=NAVY,
        edgecolor="none",
        rasterized=True,
    )
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    ax.plot([lo, hi], [lo, hi], color=RUST, lw=1.2, ls="--", label="y = x")
    metrics = regression_metrics(y_true, y_pred)
    ax.text(
        0.04,
        0.96,
        f"R²  = {metrics.r2:+.3f}\nρ  = {metrics.spearman:+.3f}\nn  = {metrics.n:,d}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        family="DejaVu Sans Mono",
        color=SLATE,
        bbox={"facecolor": CREAM, "edgecolor": SLATE, "boxstyle": "round,pad=0.3", "lw": 0.5},
    )
    ax.set_xlabel("measured log₁₀ expression", fontsize=9)
    ax.set_ylabel("predicted log₁₀ expression", fontsize=9)
    ax.set_title(title, fontsize=11, color=NAVY)
    ax.tick_params(labelsize=8)
    ax.set_facecolor(CREAM)
    for spine in ax.spines.values():
        spine.set_color(SLATE)
        spine.set_linewidth(0.6)


def make_calibration_plots(device: str) -> None:
    print("\n[calibration plots]")
    df = load_cached("urtecho2019")

    splits_random = random_split(df, val_frac=0.1, test_frac=0.1, seed=RANDOM_SEED)
    model_random = _load_cnn("random_8_1_1", device)
    pred_random = predict(model_random, splits_random["test"], seq_length=SEQ_LENGTH, device=device)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor(CREAM)
    _scatter_calibration(
        ax,
        splits_random["test"]["log_expression"].to_numpy(),
        pred_random,
        "Random split — held-out test set",
    )
    fig.tight_layout()
    out = FIG_DIR / "calibration_random.png"
    fig.savefig(out, dpi=160, facecolor=CREAM)
    plt.close(fig)
    print(f"  saved: {out}")

    splits_lm10 = leave_element_out_split(
        df, element_col="m10_pattern", val_frac=0.1, seed=RANDOM_SEED
    )
    model_lm10 = _load_cnn("leave_m10_out", device)
    pred_lm10 = predict(model_lm10, splits_lm10["test"], seq_length=SEQ_LENGTH, device=device)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    fig.patch.set_facecolor(CREAM)
    _scatter_calibration(
        ax,
        splits_lm10["test"]["log_expression"].to_numpy(),
        pred_lm10,
        f"Leave-element-out — never-seen -10 pattern: {splits_lm10['holdout_value']!r}",
    )
    fig.tight_layout()
    out = FIG_DIR / "calibration_lm10.png"
    fig.savefig(out, dpi=160, facecolor=CREAM)
    plt.close(fig)
    print(f"  saved: {out}")


def make_generalization_bar() -> None:
    print("\n[generalization_bar.png]")
    baselines = json.loads((REPORTS_DIR / "baselines_results.json").read_text())
    cnn = json.loads((REPORTS_DIR / "cnn_results.json").read_text())

    splits = ["random_8_1_1", "leave_spacer_out", "leave_m10_out"]
    split_labels = ["Random 80/10/10", "Leave-spacer-out", "Leave-m10-out"]
    models = [
        ("PWM Ridge", "pwm", "#9aa6b2"),
        ("k-mer + XGBoost", "kmer_xgb", RUST),
        ("PromoterNet CNN", "cnn", NAVY),
    ]

    x = np.arange(len(splits))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 4.0))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    for i, (label, key, color) in enumerate(models):
        values: list[float] = []
        for s in splits:
            if key == "cnn":
                values.append(float(cnn[s]["cnn"]["r2"]))
            else:
                values.append(float(baselines[s][key]["r2"]))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor=SLATE, lw=0.4)
        for bar, v in zip(bars, values, strict=False):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.02 if v >= 0 else v - 0.05,
                f"{v:.2f}",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8,
                color=SLATE,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=10, color=SLATE)
    ax.set_ylabel("test R²", fontsize=10, color=SLATE)
    ax.set_ylim(-0.1, 1.05)
    ax.axhline(0, color=SLATE, lw=0.5)
    ax.set_title(
        "PromoterNet generalizes when k-mer features cannot",
        fontsize=12,
        color=NAVY,
    )
    ax.legend(loc="lower left", fontsize=9, frameon=False)
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(SLATE)
        spine.set_linewidth(0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = FIG_DIR / "generalization_bar.png"
    fig.savefig(out, dpi=160, facecolor=CREAM)
    plt.close(fig)
    print(f"  saved: {out}")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    make_motif_logos(device)
    make_calibration_plots(device)
    make_generalization_bar()
    print("\nAll figures written to:", FIG_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
