"""Generic training loop for CNN regression on promoter sequences."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from promoternet.features import one_hot_batch


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 8
    min_delta: float = 1e-4
    seq_length: int = 150
    device: str | None = None


def _resolve_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool, seq_length: int) -> DataLoader:
    X = one_hot_batch(df["sequence"].tolist(), length=seq_length)
    y = df["log_expression"].to_numpy().astype(np.float32)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=0)


def train_cnn(
    model: nn.Module,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: TrainConfig | None = None,
    on_epoch: Callable[[int, float, float], None] | None = None,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train a CNN with MSE loss, Adam, early stopping on val MSE."""
    cfg = config or TrainConfig()
    device = _resolve_device(cfg.device)
    model = model.to(device)

    train_loader = _make_loader(train_df, cfg.batch_size, shuffle=True, seq_length=cfg.seq_length)
    val_loader = _make_loader(val_df, cfg.batch_size, shuffle=False, seq_length=cfg.seq_length)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.MSELoss()

    history = {"train_mse": [], "val_mse": []}
    best_val = float("inf")
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    epochs_without_improvement = 0

    for epoch in range(cfg.epochs):
        model.train()
        total_train_loss = 0.0
        total_train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
            total_train_n += xb.size(0)
        train_mse = total_train_loss / max(total_train_n, 1)

        model.eval()
        total_val_loss = 0.0
        total_val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total_val_loss += loss.item() * xb.size(0)
                total_val_n += xb.size(0)
        val_mse = total_val_loss / max(total_val_n, 1)

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        if on_epoch is not None:
            on_epoch(epoch, train_mse, val_mse)

        if best_val - val_mse > cfg.min_delta:
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model, history


def predict(
    model: nn.Module,
    df: pd.DataFrame,
    seq_length: int = 150,
    batch_size: int = 256,
    device: str | None = None,
) -> np.ndarray:
    dev = _resolve_device(device)
    model = model.to(dev).eval()
    loader = _make_loader(df, batch_size=batch_size, shuffle=False, seq_length=seq_length)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(dev)
            preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds, axis=0)
