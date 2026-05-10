"""Train the PromoterCNN on three split regimes and report metrics.

Mirrors scripts/run_baselines.py so results are directly comparable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from promoternet.data import (
    leave_element_out_split,
    load_cached,
    random_split,
)
from promoternet.eval import regression_metrics
from promoternet.models import PromoterCNN
from promoternet.train import TrainConfig, predict, train_cnn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SEQ_LENGTH = 150


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _train_eval(
    name: str,
    splits: dict[str, pd.DataFrame],
    cfg: TrainConfig,
) -> dict[str, dict[str, float | int]]:
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    print(f"\n=== Split: {name} ===")
    print(f"  train={len(train_df):,d}  val={len(val_df):,d}  test={len(test_df):,d}")
    if "holdout_value" in splits:
        print(f"  holdout_value: {splits['holdout_value']!r}")

    _seed_everything(RANDOM_SEED)
    model = PromoterCNN(seq_len=SEQ_LENGTH)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model parameters: {n_params:,d}")

    t0 = time.time()
    model, history = train_cnn(
        model,
        train_df,
        val_df,
        config=cfg,
        on_epoch=lambda e, t, v: print(f"    epoch {e:02d}  train_mse={t:.4f}  val_mse={v:.4f}"),
    )
    elapsed = time.time() - t0
    n_epochs = len(history["train_mse"])
    print(f"  trained {n_epochs} epochs in {elapsed:.1f}s ({elapsed / n_epochs:.2f}s/epoch)")

    pred_test = predict(model, test_df, seq_length=SEQ_LENGTH, device=cfg.device)
    test_metrics = regression_metrics(test_df["log_expression"].to_numpy(), pred_test)
    print(f"  [CNN]              {test_metrics.as_row()}")

    ckpt_path = CHECKPOINTS_DIR / f"cnn_{name}.pt"
    torch.save(model.state_dict(), ckpt_path)

    return {"cnn": test_metrics.as_dict()}


def main() -> int:
    print("Loading processed Urtecho 2019 dataset ...")
    df = load_cached("urtecho2019")
    print(f"  rows: {len(df):,d}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    cfg = TrainConfig(
        epochs=60,
        batch_size=128,
        lr=1e-3,
        weight_decay=1e-5,
        patience=8,
        seq_length=SEQ_LENGTH,
        device=device,
    )

    results: dict[str, dict[str, dict[str, float | int]]] = {}

    results["random_8_1_1"] = _train_eval(
        "random_8_1_1",
        random_split(df, val_frac=0.1, test_frac=0.1, seed=RANDOM_SEED),
        cfg,
    )
    results["leave_spacer_out"] = _train_eval(
        "leave_spacer_out",
        leave_element_out_split(df, element_col="spacer", val_frac=0.1, seed=RANDOM_SEED),
        cfg,
    )
    results["leave_m10_out"] = _train_eval(
        "leave_m10_out",
        leave_element_out_split(df, element_col="m10_pattern", val_frac=0.1, seed=RANDOM_SEED),
        cfg,
    )

    out_path = REPORTS_DIR / "cnn_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
