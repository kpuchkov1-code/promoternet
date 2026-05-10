"""Train and evaluate PWM and k-mer + XGBoost baselines on Urtecho 2019.

Reports metrics on three test regimes:
  1. Random 80/10/10 split (table-stakes).
  2. Leave-spacer-out: train on all but the largest spacer class.
  3. Leave-m10-out: train on all but the largest -10 mutation pattern.

Saves predictions and metrics to reports/ for downstream comparison with CNN.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from promoternet.data import (
    PROCESSED_DIR,
    leave_element_out_split,
    load_cached,
    random_split,
)
from promoternet.eval import regression_metrics
from promoternet.features import kmer_feature_matrix_multi, pwm_features

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
KMER_KS = (4, 5, 6)


def _fit_pwm(train_df: pd.DataFrame) -> Ridge:
    X = pwm_features(train_df["sequence"].tolist())
    y = train_df["log_expression"].to_numpy()
    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X, y)
    return model


def _predict_pwm(model: Ridge, df: pd.DataFrame) -> np.ndarray:
    X = pwm_features(df["sequence"].tolist())
    return model.predict(X)


def _fit_kmer_xgb(train_df: pd.DataFrame, val_df: pd.DataFrame) -> XGBRegressor:
    X_train = kmer_feature_matrix_multi(train_df["sequence"].tolist(), KMER_KS)
    y_train = train_df["log_expression"].to_numpy()
    X_val = kmer_feature_matrix_multi(val_df["sequence"].tolist(), KMER_KS)
    y_val = val_df["log_expression"].to_numpy()

    model = XGBRegressor(
        n_estimators=600,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.6,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        early_stopping_rounds=30,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _predict_kmer_xgb(model: XGBRegressor, df: pd.DataFrame) -> np.ndarray:
    X = kmer_feature_matrix_multi(df["sequence"].tolist(), KMER_KS)
    return model.predict(X)


def evaluate_split(
    name: str, splits: dict[str, pd.DataFrame]
) -> dict[str, dict[str, float | int]]:
    train_df = splits["train"]
    val_df = splits["val"]
    test_df = splits["test"]

    print(f"\n=== Split: {name} ===")
    print(f"  train={len(train_df):,d}  val={len(val_df):,d}  test={len(test_df):,d}")
    if "holdout_value" in splits:
        print(f"  holdout_value: {splits['holdout_value']!r}")

    pwm_model = _fit_pwm(train_df)
    pwm_pred_test = _predict_pwm(pwm_model, test_df)
    pwm_metrics = regression_metrics(test_df["log_expression"].to_numpy(), pwm_pred_test)
    print(f"  [PWM]              {pwm_metrics.as_row()}")

    kmer_model = _fit_kmer_xgb(train_df, val_df)
    kmer_pred_test = _predict_kmer_xgb(kmer_model, test_df)
    kmer_metrics = regression_metrics(test_df["log_expression"].to_numpy(), kmer_pred_test)
    print(f"  [k-mer XGBoost]    {kmer_metrics.as_row()}")

    return {
        "pwm": pwm_metrics.as_dict(),
        "kmer_xgb": kmer_metrics.as_dict(),
    }


def main() -> int:
    print("Loading processed Urtecho 2019 dataset ...")
    df = load_cached("urtecho2019")
    print(f"  loaded {len(df):,d} rows from {PROCESSED_DIR}")

    results: dict[str, dict[str, dict[str, float | int]]] = {}

    results["random_8_1_1"] = evaluate_split(
        "random_8_1_1",
        random_split(df, val_frac=0.1, test_frac=0.1, seed=RANDOM_SEED),
    )

    results["leave_spacer_out"] = evaluate_split(
        "leave_spacer_out",
        leave_element_out_split(df, element_col="spacer", val_frac=0.1, seed=RANDOM_SEED),
    )

    results["leave_m10_out"] = evaluate_split(
        "leave_m10_out",
        leave_element_out_split(df, element_col="m10_pattern", val_frac=0.1, seed=RANDOM_SEED),
    )

    out_path = REPORTS_DIR / "baselines_results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
