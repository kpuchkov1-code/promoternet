"""Evaluation metrics for promoter strength regression."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class RegressionMetrics:
    n: int
    r2: float
    pearson: float
    spearman: float
    mse: float
    mae: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "r2": round(self.r2, 4),
            "pearson": round(self.pearson, 4),
            "spearman": round(self.spearman, 4),
            "mse": round(self.mse, 4),
            "mae": round(self.mae, 4),
        }

    def as_row(self) -> str:
        return (
            f"n={self.n:,d}  R2={self.r2:+.4f}  "
            f"Pearson={self.pearson:+.4f}  Spearman={self.spearman:+.4f}  "
            f"MSE={self.mse:.4f}  MAE={self.mae:.4f}"
        )


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    if y_true.size < 2:
        raise ValueError("need at least 2 points for correlation metrics")

    pearson_r = float(pearsonr(y_true, y_pred).statistic)
    spearman_r = float(spearmanr(y_true, y_pred).statistic)
    return RegressionMetrics(
        n=int(y_true.size),
        r2=float(r2_score(y_true, y_pred)),
        pearson=pearson_r,
        spearman=spearman_r,
        mse=float(mean_squared_error(y_true, y_pred)),
        mae=float(mean_absolute_error(y_true, y_pred)),
    )
