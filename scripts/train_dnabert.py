"""Fine-tune DNABERT-6 on Urtecho 2019, three split regimes.

DNABERT-2-117M's Mosaic-BERT requires Triton/flash-attn that don't build cleanly
on Windows; Nucleotide Transformer v2's bundled config breaks against
transformers >= 5.x. DNABERT-6 (Ji, Zhou et al, same lab as DNABERT-2) uses
standard transformers BertModel — drops in cleanly, fits in 8 GB VRAM with
batch=16 + AMP, and is the natural foundation-model baseline for prokaryotic
sequences on this hardware.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from transformers import BertTokenizer

from promoternet.data import (
    leave_element_out_split,
    load_cached,
    random_split,
)
from promoternet.eval import regression_metrics
from promoternet.nt_model import (
    NT_MODEL_ID,
    estimate_token_length,
    predict_nt,
    train_nt,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42


def _train_eval(name: str, splits: dict, max_length: int) -> dict:
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    print(f"\n=== Split: {name} ===")
    print(f"  train={len(train_df):,d}  val={len(val_df):,d}  test={len(test_df):,d}")
    if "holdout_value" in splits:
        print(f"  holdout_value: {splits['holdout_value']!r}")
    print(f"  max_length (tokens): {max_length}")

    t0 = time.time()
    model, tokenizer, history = train_nt(
        train_df,
        val_df,
        epochs=8,
        batch_size=16,
        lr=3e-5,
        weight_decay=0.01,
        patience=2,
        max_length=max_length,
        use_amp=True,
        on_epoch=lambda e, t, v: print(f"    epoch {e:02d}  train_mse={t:.4f}  val_mse={v:.4f}"),
    )
    elapsed = time.time() - t0
    n_epochs = len(history["train_mse"])
    print(f"  trained {n_epochs} epochs in {elapsed:.1f}s ({elapsed / n_epochs:.1f}s/epoch)")

    pred_test = predict_nt(model, tokenizer, test_df, batch_size=32, max_length=max_length)
    metrics = regression_metrics(test_df["log_expression"].to_numpy(), pred_test)
    print(f"  [DNABERT-6]        {metrics.as_row()}")

    ckpt_path = CHECKPOINTS_DIR / f"nt_{name}.pt"
    torch.save(model.state_dict(), ckpt_path)
    return {"nt": metrics.as_dict()}


def main() -> int:
    print(f"Loading dataset + tokenizer ({NT_MODEL_ID}) ...")
    df = load_cached("urtecho2019")
    tokenizer = BertTokenizer.from_pretrained(NT_MODEL_ID)
    max_length = estimate_token_length(df["sequence"], tokenizer, sample=500)
    print(f"  rows: {len(df):,d}  empirical max tokens (p99): {max_length}")
    max_length = min(max(max_length, 156), 200)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  gpu: {gpu_name} ({gpu_mem:.1f} GB)")

    results: dict = {}
    results["random_8_1_1"] = _train_eval(
        "random_8_1_1",
        random_split(df, val_frac=0.1, test_frac=0.1, seed=RANDOM_SEED),
        max_length,
    )
    results["leave_spacer_out"] = _train_eval(
        "leave_spacer_out",
        leave_element_out_split(df, element_col="spacer", val_frac=0.1, seed=RANDOM_SEED),
        max_length,
    )
    results["leave_m10_out"] = _train_eval(
        "leave_m10_out",
        leave_element_out_split(df, element_col="m10_pattern", val_frac=0.1, seed=RANDOM_SEED),
        max_length,
    )

    out_path = REPORTS_DIR / "nt_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
