# PromoterNet — benchmark results

Dataset: Urtecho et al 2019 (Biochemistry), 10,898 σ70 promoter variants from a combinatorial library spanning 3 UP elements × 8 -35 elements × 8 spacers × 8 -10 elements × 8 backgrounds. Expression measured by RNA-seq / DNA-seq ratio in MOPS + glucose, reported per-variant as `RNA_exp_average`. Target = log₁₀(RNA_exp_average).

All sequences are exactly 150 bp. After filtering for ACGT-only sequences, positive expression, and ≥3 integrated barcodes, 10,898 variants remain.

## Models

| Model | Featurization | Algorithm | Parameters |
|---|---|---|---|
| PWM Ridge | -10 / -35 PWM scan, top score + position (4 features) | Ridge (α=1.0) | 5 |
| k-mer + XGBoost | counts of k = 4, 5, 6 (4,160 features) | XGBoost (600 trees, depth 7) | ≈100k effective |
| PromoterNet CNN | one-hot (4, 150) | 2× Conv1d + BN + MaxPool + FC | 129,857 |

## Test-set R² across three split regimes

| Split | n_train / n_val / n_test | PWM Ridge | k-mer + XGBoost | PromoterNet CNN |
|---|---|---|---|---|
| Random 80/10/10 | 8,718 / 1,090 / 1,090 | **0.18** | **0.96** | **0.95** |
| Leave-spacer-out (held out: `ECK125136938`, n=1,463) | 8,491 / 944 / 1,463 | **0.23** | **0.85** | **0.87** |
| Leave-m10-out (held out: `7T->A`, n=1,376) | 8,570 / 952 / 1,376 | **0.13** | **0.23** | **0.67** |

## Spearman rank correlation (what matters for ranking promoters)

| Split | PWM Ridge | k-mer + XGBoost | PromoterNet CNN |
|---|---|---|---|
| Random 80/10/10 | 0.36 | 0.83 | 0.80 |
| Leave-spacer-out | 0.40 | 0.71 | 0.71 |
| Leave-m10-out | 0.34 | 0.67 | **0.84** |

## Headline finding

On the random split, k-mer counts can shortcut the prediction by memorizing which -10 element each variant uses (because k-mers are positional-bag-of-words and the library is enumerated combinatorially). The CNN matches that performance. **Where the two diverge is the leave-element-out test**: when an entire -10 mutation pattern is held out from training, the k-mer model collapses to R²=0.23 because it has never seen the discriminative k-mers; the CNN holds at R²=0.67 (Spearman 0.84) because it has learned to attend to the -10 *position* rather than memorize -10 *identity*.

For Neobe, this is the deployment-relevant scenario: the team wants to engineer new promoter parts whose specific -10 / -35 / spacer composition was not in their characterization data. A model that only works on memorized parts has no value over the wet lab.

## Interpretability

Gradient saliency on the top-200 highest-expression test promoters concentrates at positions {87, 89} and {108, 109, 113}, which correspond to the -35 box and -10 box positions in the Urtecho library design.

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
python scripts\saliency_analysis.py
python scripts\make_figures.py
pytest tests/
```

Hardware used: NVIDIA RTX 3070 Laptop (8 GB VRAM). Each CNN training run completes in ~10 seconds; full pipeline including all baselines completes in under 5 minutes after data download.
