# PromoterNet

ML benchmark that predicts *E. coli* σ70 promoter strength from a 150 bp sequence, comparing PWM, k-mer + XGBoost, a 1D CNN, and a DNABERT-6 fine-tune — with an emphasis on **generalization to held-out promoter elements**, not just random-split accuracy.

The motivating application is **tumor-selective expression** of payloads in engineered live biotherapeutic products (LBPs). Live bacterial cancer therapies require payload expression to be high inside the tumor and silent in healthy tissue — a selectivity problem solved with sensor-driven promoter circuits whose dynamic range is the IND-critical safety parameter. A sequence-to-strength model that generalizes to *new* promoter elements (rather than memorizing characterized ones) reduces the wet-lab combinatorics of designing those circuits.

## Approach

1. **Dataset.** Urtecho et al 2019 (*Biochemistry*) — 10,898 σ70 promoter variants from a combinatorial library (3 UP × 8 -35 × 8 spacers × 8 -10 × 8 backgrounds), all exactly 150 bp, expression measured by RNA-seq / DNA-seq ratio. Publicly hosted at `github.com/KosuriLab`.
2. **Features.** Position weight matrix (PWM) on the σ70 -10 / -35 motifs; k-mer counts (k = 4, 5, 6); one-hot encoding for the CNN; overlapping 6-mer tokens for DNABERT-6.
3. **Models.** PWM Ridge baseline, gradient-boosted k-mer regression (XGBoost), DeepBind/Basset-style 1D CNN, and a DNABERT-6 fine-tune with a regression head.
4. **Evaluation — three split regimes:**
   - **Random 80/10/10** — table-stakes.
   - **Leave-spacer-out** — an entire spacer class held out of training.
   - **Leave-m10-out** — an entire -10 mutation pattern held out of training. *This is the deployment-relevant test:* a design tool must extrapolate to elements it has never characterized.
5. **Interpretability.** Gradient saliency on the trained CNN aggregated across high-expression promoters, plus first-layer convolutional filter analysis, confirming unsupervised rediscovery of the canonical -10 (`TATAAT`) and -35 (`TTGACA`) boxes.

## Headline findings

| Split | PWM | k-mer XGB | CNN | DNABERT-6 |
|---|---|---|---|---|
| Random 80/10/10 | 0.18 | **0.96** | **0.95** | **0.95** |
| Leave-spacer-out | 0.23 | 0.85 | 0.87 | **0.87** |
| Leave-m10-out | 0.13 | 0.23 | **0.67** | 0.12 |

*(test R² on Urtecho 2019. Full table incl. Spearman in [`reports/results_table.md`](reports/results_table.md).)*

1. **The k-mer trap.** The k-mer model scores 0.96 on a random split but collapses to 0.23 when forced to predict for a -10 element it has never seen — it was memorizing element *identity*, not learning σ70 *grammar*.
2. **The CNN closes the gap.** It holds at R²=0.67 (Spearman 0.84) on leave-m10-out because its filters learn positional patterns over the -35 / -10 regions. Those filters rediscover `TATAAT` and `TTGACA` without supervision.
3. **Foundation model: cheap ranking, biased calibration.** DNABERT-6 ranks well (Spearman 0.73 on leave-m10-out) but its absolute R² collapses to 0.12 — pretrained priors don't transfer cleanly to *E. coli* σ70 context. **Use a foundation model for ranking; use a small purpose-built model when absolute calibration matters.**

## Repository layout

```
promoternet/
├── README.md                          this file
├── pyproject.toml                     Python project metadata
├── data/
│   ├── raw/                           downloaded Urtecho supplementary table
│   ├── interim/                       cleaned CSV
│   └── processed/                     unified parquet for training
├── src/promoternet/
│   ├── data.py                        loaders + splits (random / leave-spacer / leave-m10)
│   ├── features.py                    one-hot, k-mer, PWM utilities
│   ├── models.py                      PromoterCNN definition
│   ├── nt_model.py                    DNABERT-6 regressor + fine-tune loop
│   ├── train.py                       CNN training loop + early stopping
│   ├── eval.py                        R², Spearman, calibration plots
│   └── interpret.py                   gradient saliency + motif aggregation
├── scripts/
│   ├── download_data.py               fetch the Urtecho dataset
│   ├── process_urtecho.py             clean + build processed parquet
│   ├── run_baselines.py               PWM + k-mer XGBoost
│   ├── train_cnn.py                   CNN training entry point
│   ├── train_dnabert.py              DNABERT-6 fine-tune entry point
│   ├── saliency_analysis.py           saliency + motif rediscovery
│   └── make_figures.py                generate all benchmark figures
├── notebooks/                         EDA + analysis notebooks
├── tests/                             pytest unit tests
└── reports/
    ├── results_table.md               final benchmark table
    └── figures/                       saliency, calibration, generalization plots
```

## Getting started

```powershell
# Windows PowerShell
cd promoternet
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

Hardware used: NVIDIA RTX 3070 Laptop (8 GB VRAM). CNN trains in ~10 s/run; DNABERT-6 fine-tune ~7–40 min/split.

## Citations

- Urtecho G, Tripp AD, Insigne KD, Kim H, Kosuri S. *Systematic Dissection of Sequence Elements Controlling σ70 Promoters Using a Genomically Encoded Multiplexed Reporter Assay in Escherichia coli.* Biochemistry 58(11): 1539–1551, 2019.
- Ji Y, Zhou Z, Liu H, Davuluri RV. *DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome.* Bioinformatics 37(15): 2112–2120, 2021.
- Alipanahi B, Delong A, Weirauch MT, Frey BJ. *Predicting the sequence specificities of DNA- and RNA-binding proteins by deep learning (DeepBind).* Nat Biotechnol 33: 831–838, 2015.
