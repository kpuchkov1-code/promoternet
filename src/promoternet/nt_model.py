"""DNABERT-6 fine-tuning with a regression head.

DNABERT-2-117M's Mosaic-BERT architecture depends on Triton/flash-attn kernels
that don't have a clean Windows build, and Nucleotide Transformer v2's bundled
config breaks against transformers >= 5.x. DNABERT-6 from the same Zhihan
Zhou / Liu lab uses standard ``transformers.BertModel`` — same lineage, but it
works on the RTX 3070 Laptop's 8 GB VRAM with no hacks.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel, BertTokenizer

NT_MODEL_ID = "zhihan1996/DNA_bert_6"
KMER_LEN = 6


def _seq_to_kmers(seq: str, k: int = KMER_LEN) -> str:
    return " ".join(seq[i : i + k] for i in range(len(seq) - k + 1))


class _PromoterTokenDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        labels: list[float],
        tokenizer: BertTokenizer,
        max_length: int,
    ):
        self.kmer_strings = [_seq_to_kmers(s) for s in sequences]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.kmer_strings)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.kmer_strings[idx],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


class NucleotideTransformerRegressor(nn.Module):
    def __init__(self, model_id: str = NT_MODEL_ID, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone = BertModel.from_pretrained(model_id)
        hidden = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(pooled).squeeze(-1)


def _make_loader(
    df: pd.DataFrame,
    tokenizer: BertTokenizer,
    batch_size: int,
    shuffle: bool,
    max_length: int,
) -> DataLoader:
    ds = _PromoterTokenDataset(
        sequences=df["sequence"].tolist(),
        labels=df["log_expression"].astype(float).tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_nt(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    epochs: int = 6,
    batch_size: int = 16,
    lr: float = 3e-5,
    weight_decay: float = 0.01,
    patience: int = 2,
    max_length: int = 156,
    use_amp: bool = True,
    on_epoch=None,
) -> tuple[NucleotideTransformerRegressor, BertTokenizer, dict[str, list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained(NT_MODEL_ID)
    model = NucleotideTransformerRegressor().to(device)

    train_loader = _make_loader(train_df, tokenizer, batch_size, True, max_length)
    val_loader = _make_loader(val_df, tokenizer, batch_size, False, max_length)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    history = {"train_mse": [], "val_mse": []}
    best_val = float("inf")
    best_state = None
    epochs_without = 0

    for epoch in range(epochs):
        model.train()
        total_loss, total_n = 0.0, 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                preds = model(input_ids, attention_mask)
                loss = loss_fn(preds, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * input_ids.size(0)
            total_n += input_ids.size(0)
        train_mse = total_loss / max(total_n, 1)

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["label"].to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    preds = model(input_ids, attention_mask)
                    loss = loss_fn(preds, labels)
                val_loss += loss.item() * input_ids.size(0)
                val_n += input_ids.size(0)
        val_mse = val_loss / max(val_n, 1)

        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        if on_epoch is not None:
            on_epoch(epoch, train_mse, val_mse)

        if val_mse < best_val - 1e-4:
            best_val = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without = 0
        else:
            epochs_without += 1
            if epochs_without >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, tokenizer, history


def predict_nt(
    model: NucleotideTransformerRegressor,
    tokenizer: BertTokenizer,
    df: pd.DataFrame,
    batch_size: int = 32,
    max_length: int = 156,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    loader = _make_loader(df, tokenizer, batch_size, False, max_length)
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                p = model(input_ids, attention_mask)
            preds.append(p.float().cpu().numpy())
    return np.concatenate(preds, axis=0)


def estimate_token_length(sequences: Iterable[str], tokenizer, sample: int = 200) -> int:
    seqs = list(sequences)[:sample]
    lens = [
        len(tokenizer.encode(_seq_to_kmers(s), add_special_tokens=True)) for s in seqs
    ]
    return int(np.percentile(lens, 99))
