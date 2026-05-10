# PromoterNet

ML model that predicts E. coli promoter strength from 50–150 bp upstream DNA sequence, benchmarking PWM, k-mer + XGBoost, 1D CNN, and DNABERT-2 fine-tuning across three published massively parallel reporter assay (MPRA) libraries.

The motivating application is **tumor-selective expression** of payloads in engineered live biotherapeutic products (LBPs). Live bacterial cancer therapies require payload expression to be high inside the tumor and silent in healthy tissue — a selectivity problem that is currently solved with sensor-driven promoter circuits whose dynamic range is the IND-critical safety parameter. A sequence-to-strength model that generalizes across promoter libraries reduces the wet-lab combinatorics for designing those circuits.

## Approach

1. **Datasets.** Three published bacterial promoter MPRA libraries:
   - Kosuri et al 2013 PNAS — 12,563 σ70 promoter variants (FACS-seq).
   - Höllerer et al 2020 Nat Comm — synthetic promoter library, MPRA.
   - LaFleur et al 2022 Nat Comm — Salis lab in-vivo promoter calculator data.
2. **Features.** Position weight matrix (PWM) on σ70 -10 / -35 motifs; k-mer counts (k=4,5,6); one-hot encoding for the CNN; BPE tokens for DNABERT-2.
3. **Models.** PWM regression baseline, gradient-boosted k-mer regression (XGBoost), DeepBind-style 1D CNN, DNABERT-6 fine-tune with a regression head.
4. **Evaluation.**
   - Random 80/10/10 split — table-stakes.
   - **Leave-library-out** — train on Kosuri, evaluate on Höllerer / LaFleur. Tests whether the model learned σ70 transcription rather than memorized one library's sequence biases.
5. **Interpretability.** Gradient saliency on the trained CNN aggregated across high-expression promoters, with a sequence logo at attended positions to confirm rediscovery of the canonical -10 (TATAAT) and -35 (TTGACA) boxes.

## Why this matters for tumor-selective expression in LBPs

Expression-level engineering at IND is a *budget allocation* across three layers: sensor specificity, promoter strength under condition, and ribosome-binding-site (RBS) translation efficiency. A predictive model for the middle layer (promoter strength as a function of sequence) lets the engineering team (a) rank candidate hypoxia/lactate-induced promoter parts in silico before synthesis and (b) propose tuned variants with target dynamic-range characteristics. Empirically, in-silico ranking has cut MPRA candidate counts by an order of magnitude in published applications (LaFleur 2022). The benchmark below establishes the floor on what an out-of-the-box model achieves and where the gap to a Neobe-internal model must be closed with proprietary characterization data.

## Repository layout

```
promoternet/
├── README.md                          this file
├── pyproject.toml                     Python project metadata
├── data/
│   ├── raw/                           downloaded supplementary tables
│   ├── interim/                       cleaned per-library CSVs
│   └── processed/                     unified parquet for training
├── src/promoternet/
│   ├── data.py                        loaders + splits (random + leave-library-out)
│   ├── features.py                    one-hot, k-mer, PWM utilities
│   ├── models.py                      PWM, XGBoost, CNN, DNABERT-2 wrappers
│   ├── train.py                       training loop + early stopping
│   ├── eval.py                        R², Spearman, calibration plots
│   └── interpret.py                   gradient saliency + motif aggregation
├── scripts/
│   ├── download_data.py               fetch all three datasets
│   ├── run_baselines.py               PWM + k-mer XGBoost
│   ├── train_cnn.py                   CNN training entry point
│   ├── train_dnabert.py               DNABERT-2 fine-tune entry point
│   └── leave_library_out_eval.py      cross-library generalization
├── notebooks/                         EDA + analysis notebooks
├── tests/                             pytest unit tests
└── reports/
    ├── results_table.md               final benchmark table
    ├── neobe_selectivity_memo.md      one-page applicability memo
    └── figures/                       saliency, calibration, generalization plots
```

## Getting started

```powershell
# Windows PowerShell
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

## Citations

- Kosuri S, Goodman DB, Cambray G, Mutalik VK, Gallup M, Endy D, Mutalik VK, Arkin AP. *Composability of regulatory sequences controlling transcription and translation in Escherichia coli.* PNAS 110(34): 14024–14029, 2013.
- Höllerer S, Papaxanthos L, Gumpinger AC, Fischer K, Beisel C, Borgwardt K, Benenson Y, Jeschek M. *Large-scale DNA-based phenotypic recording and deep learning enable highly accurate sequence-function mapping.* Nat Commun 11: 3551, 2020.
- LaFleur TL, Hossain A, Salis HM. *Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria.* Nat Commun 13: 5159, 2022.
- Vaishnav ED et al. *The evolution, evolvability and engineering of gene regulatory DNA.* Nature 603: 455–463, 2022.
- Ji Y, Zhou Z, Liu H, Davuluri RV. *DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome.* Bioinformatics 37(15): 2112–2120, 2021.
