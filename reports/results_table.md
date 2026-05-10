# PromoterNet — benchmark results

Dataset: Urtecho et al 2019 (Biochemistry), 10,898 σ70 promoter variants from a combinatorial library spanning 3 UP elements × 8 -35 elements × 8 spacers × 8 -10 elements × 8 backgrounds. Expression measured by RNA-seq / DNA-seq ratio in MOPS + glucose, reported per-variant as `RNA_exp_average`. Target = log₁₀(RNA_exp_average).

All sequences are exactly 150 bp. After filtering for ACGT-only sequences, positive expression, and ≥3 integrated barcodes, 10,898 variants remain.

## Models

| Model | Featurization | Algorithm | Parameters |
|---|---|---|---|
| PWM Ridge | -10 / -35 PWM scan, top score + position (4 features) | Ridge (α=1.0) | 5 |
| k-mer + XGBoost | counts of k = 4, 5, 6 (4,160 features) | XGBoost (600 trees, depth 7) | ≈100k effective |
| PromoterNet CNN | one-hot (4, 150) | 2× Conv1d + BN + MaxPool + FC | 129,857 |
| DNABERT-6 fine-tune | 6-mer overlapping tokenization | BertModel + MLP head, AMP fp16 | 89.2 M |

DNABERT-6 model: `zhihan1996/DNA_bert_6` (Ji et al 2021, *Bioinformatics*) loaded via `transformers.BertModel`. Trained head + backbone with AdamW, lr=3e-5, batch=16, weight decay=0.01, gradient clip=1.0, AMP mixed precision, early stopping on validation MSE (patience=2).

## Test-set R² across three split regimes

| Split | n_train / n_val / n_test | PWM Ridge | k-mer + XGBoost | PromoterNet CNN | DNABERT-6 |
|---|---|---|---|---|---|
| Random 80/10/10 | 8,718 / 1,090 / 1,090 | 0.18 | **0.96** | **0.95** | **0.95** |
| Leave-spacer-out (held out: `ECK125136938`, n=1,463) | 8,491 / 944 / 1,463 | 0.23 | 0.85 | 0.87 | **0.87** |
| Leave-m10-out (held out: `7T->A`, n=1,376) | 8,570 / 952 / 1,376 | 0.13 | 0.23 | **0.67** | 0.12 |

## Spearman rank correlation (what matters for ranking promoters)

| Split | PWM Ridge | k-mer + XGBoost | PromoterNet CNN | DNABERT-6 |
|---|---|---|---|---|
| Random 80/10/10 | 0.36 | 0.83 | 0.80 | 0.83 |
| Leave-spacer-out | 0.40 | 0.71 | 0.71 | 0.73 |
| Leave-m10-out | 0.34 | 0.67 | **0.84** | 0.73 |

## Headline findings

**1. The k-mer trap.** On the random split, k-mer counts can shortcut the prediction by memorizing which -10 element each variant uses (the library is enumerated combinatorially). When an entire -10 mutation pattern is held out, the k-mer model collapses to R²=0.23. It was memorizing -10 *identity*, not learning σ70 *grammar*.

**2. The CNN closes the generalization gap.** On the same leave-m10-out test, the purpose-built 1D CNN holds at R²=0.67 (Spearman 0.84) because it has learned positional attention over the -35 / -10 regions rather than memorizing element identity. Convolutional filters in the first layer recover the canonical TATAAT and TTGACA motifs without supervision (cosine similarity 0.91 / 0.82 against consensus).

**3. Foundation models give cheap ranking but biased calibration.** DNABERT-6 fine-tune matches CNN/k-mer on the easy splits (R²=0.95, 0.87) and *exceeds* both on Spearman for leave-spacer-out (0.73). But on leave-m10-out, DNABERT-6 collapses to R²=0.12 while keeping Spearman at 0.73 — meaning it ranks promoters correctly but predicts the wrong absolute log-expression by a constant offset. The pretrained priors do not perfectly transfer to E. coli σ70 promoter context, and 6 epochs of regression-head training cannot fully correct the bias. **For ranking-mode applications (\"give me the top 20 candidates\"), DNABERT-6 is fine. For calibration-mode applications (\"predict the absolute ON/OFF dynamic range so we can budget IND-grade leakage\"), the small purpose-built CNN is the right model.**

This finding is the most useful from a deployment perspective — knowing *when* to reach for a foundation model vs a small purpose-built model is the practical question for any biotech ML adoption.

## Interpretability — sigma70 motif rediscovery

Gradient saliency on the top-200 highest-expression test promoters concentrates at positions {87, 89} and {108, 109, 113}, corresponding to the -35 box and -10 box positions in the Urtecho library design.

Convolutional filters in the first layer rediscover the canonical σ70 motifs without supervision:

| Filter | Cosine to consensus | Top-positional motif |
|---|---|---|
| 55 | 0.91 vs TATAAT | `CAAGTATAATTA` |
| 54 | 0.88 vs TATAAT | `AAATATATTGGT` |
| 4  | 0.85 vs TATAAT | `AAAAATACTGGC` |
| 37 | 0.82 vs TTGACA | `ATGTACTTGTCA` |
| 18 | 0.80 vs TTGACA | `TACCGTGACACT` |
| 63 | 0.80 vs TTGACA | `TTATGTTGAAAT` |

## Reproducibility

```powershell
cd C:\Users\Kirill\promoternet
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python scripts\download_data.py
python scripts\process_urtecho.py
python scripts\run_baselines.py
python scripts\train_cnn.py
python scripts\train_dnabert.py
python scripts\saliency_analysis.py
python scripts\make_figures.py
pytest tests/
```

Hardware: NVIDIA RTX 3070 Laptop (8 GB VRAM). CNN: ~10 seconds per training run. DNABERT-6: ~7-40 minutes per split depending on thermal headroom (89M-param fine-tune at fp16, batch=16, 145-token sequences).
