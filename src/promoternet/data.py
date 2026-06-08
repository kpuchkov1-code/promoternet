"""Data loaders for bacterial promoter MPRA datasets.

Primary dataset: Urtecho et al 2019, combinatorial σ70 promoter library.
Each variant is composed from discrete element classes, which we parse out of
the name field so we can build leave-element-out generalization splits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass(frozen=True)
class PromoterRecord:
    name: str
    sequence: str
    expression: float
    up_class: str
    m35_pattern: str
    spacer: str
    m10_pattern: str
    background: str


_UP_PREFIXES: tuple[str, ...] = ("noUP", "gourse_136fold_up", "gourse_326fold_up")
_BG_PATTERN = re.compile(r"bg\d+:\d+$")
_SHIFT_PATTERN = re.compile(r"shift_\w+_\d+:\d+$")


def _parse_name(name: str) -> tuple[str, str, str, str, str] | None:
    """Decompose an Urtecho variant name into (UP, -35, spacer, -10, background).

    Returns None if the name cannot be parsed (e.g., neg_controls).
    """
    if name.startswith("neg_control"):
        return None

    up_class = next((p for p in _UP_PREFIXES if name.startswith(p)), None)
    if up_class is None:
        return None
    rest = name[len(up_class) + 1 :]

    bg_match = _BG_PATTERN.search(rest) or _SHIFT_PATTERN.search(rest)
    if bg_match is None:
        return None
    background = bg_match.group(0)
    middle = rest[: bg_match.start()].rstrip("_")

    parts = middle.split("_")
    if len(parts) != 3:
        return None
    m35, spacer, m10 = parts
    return up_class, m35, spacer, m10, background


def load_urtecho_2019(raw_dir: Path = RAW_DIR / "urtecho2019") -> pd.DataFrame:
    """Load + join Urtecho 2019 σ70 promoter library data.

    Returns a tidy DataFrame with one row per variant:
        name, sequence, expression, log_expression,
        up_class, m35_pattern, spacer, m10_pattern, background
    """
    variants_path = raw_dir / "variant_statistics.txt"
    expression_path = raw_dir / "revLP5_Min_MOPS_glu_expression.txt"

    variants = pd.read_csv(
        variants_path,
        sep="\t",
        header=0,
        usecols=["variant", "name"],
        dtype=str,
    ).dropna()
    variants = variants.rename(columns={"variant": "sequence"})

    expression = pd.read_csv(
        expression_path,
        sep=r"\s+",
        header=0,
        usecols=["name", "RNA_exp_average", "DNA_sum", "num_barcodes_integrated"],
    )
    expression = expression.rename(columns={"RNA_exp_average": "expression"})

    df = variants.merge(expression, on="name", how="inner")

    df["sequence"] = df["sequence"].str.upper().str.strip()
    df = df[df["sequence"].str.fullmatch(r"[ACGT]+")]
    df = df[df["expression"].notna() & (df["expression"] > 0)]
    df = df[df["num_barcodes_integrated"] >= 3]

    parsed = df["name"].apply(_parse_name)
    keep_mask = parsed.notna()
    df = df.loc[keep_mask].copy()
    df[["up_class", "m35_pattern", "spacer", "m10_pattern", "background"]] = (
        pd.DataFrame(parsed[keep_mask].tolist(), index=df.index)
    )

    df["log_expression"] = np.log10(df["expression"].astype(float))

    df = df.reset_index(drop=True)
    return df[
        [
            "name",
            "sequence",
            "expression",
            "log_expression",
            "up_class",
            "m35_pattern",
            "spacer",
            "m10_pattern",
            "background",
        ]
    ]


def random_split(
    df: pd.DataFrame,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """80/10/10 random split. Returns dict with keys train/val/test."""
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(round(test_frac * len(df)))
    n_val = int(round(val_frac * len(df)))
    test_idx = idx[:n_test]
    val_idx = idx[n_test : n_test + n_val]
    train_idx = idx[n_test + n_val :]
    return {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def leave_element_out_split(
    df: pd.DataFrame,
    element_col: str,
    holdout_value: str | None = None,
    val_frac: float = 0.1,
    seed: int = 0,
) -> dict[str, pd.DataFrame]:
    """Train on all variants except those matching holdout_value in element_col.

    If holdout_value is None, picks the largest element class as the holdout
    (most rigorous test: "model has never seen this element").
    """
    if element_col not in df.columns:
        raise KeyError(f"element_col {element_col!r} not in df.columns={list(df.columns)}")

    if holdout_value is None:
        holdout_value = df[element_col].value_counts().idxmax()

    test_mask = df[element_col] == holdout_value
    test_df = df[test_mask].reset_index(drop=True)
    pool = df[~test_mask].reset_index(drop=True)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))
    n_val = int(round(val_frac * len(pool)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    return {
        "train": pool.iloc[train_idx].reset_index(drop=True),
        "val": pool.iloc[val_idx].reset_index(drop=True),
        "test": test_df,
        "holdout_value": holdout_value,  # type: ignore[dict-item]
    }


def cache_processed(df: pd.DataFrame, name: str = "urtecho2019") -> Path:
    """Write the tidy DataFrame to data/processed/<name>.parquet."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


def load_cached(name: str = "urtecho2019") -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / f"{name}.parquet")
