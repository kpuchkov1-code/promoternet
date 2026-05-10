# Memo — computational selectivity engineering for an LBP at IND

**To:** Pedro Correa de Sampaio
**From:** Kirill Puchkov
**Re:** Where I would slot in on the regulatory-DNA / sensor-circuit layer
**Date:** 2026-05-10

---

## What I built before this meeting

`PromoterNet` — a sequence-to-strength regression model trained on the Urtecho et al 2019 σ70 promoter library (10,898 variants from a combinatorial design spanning 8 -35 elements, 8 spacers, 8 -10 elements, 3 UP elements, 8 backgrounds). Source, tests, figures, and reproducibility instructions are in the project repo. ~5 days of focused work.

Four model classes benchmarked:

1. **PWM Ridge** — mechanistic baseline scoring the canonical -10 (TATAAT) and -35 (TTGACA) motifs.
2. **k-mer + XGBoost** — 4,160 k-mer features, gradient-boosted trees with early stopping.
3. **PromoterNet CNN** — DeepBind/Basset-style 1D CNN, 130k parameters, batch-norm + dropout.
4. **DNABERT-6 fine-tune** — 89M-parameter pretrained DNA BERT, regression head + backbone fine-tune in fp16.

Three evaluation regimes:

1. **Random 80/10/10** — table-stakes.
2. **Leave-spacer-out** — held-out spacer class never seen in training.
3. **Leave-m10-out** — held-out -10 mutation pattern never seen in training. *This is the deployment-relevant test*: design tools must extrapolate to parts you have not yet characterized.

## What the result actually means for the platform

| Split | PWM | k-mer XGB | CNN | DNABERT-6 |
|---|---|---|---|---|
| Random | 0.18 | **0.96** | **0.95** | **0.95** |
| Leave-spacer-out | 0.23 | 0.85 | 0.87 | **0.87** |
| Leave-m10-out | 0.13 | **0.23** | **0.67** | 0.12 |

(R² on test set, Urtecho 2019. Spearman on leave-m10-out: PWM 0.34, k-mer 0.67, CNN 0.84, DNABERT-6 0.73.)

Two findings worth Pedro's attention:

**The k-mer trap.** The k-mer model that scored 0.96 on a random split collapses to 0.23 when forced to predict for a -10 pattern it has never seen — it was memorizing -10 identity, not learning σ70 grammar. The CNN holds at 0.67 (Spearman 0.84) because it learns positional attention over the -35 / -10 regions. Convolutional filters in the first layer recover TATAAT and TTGACA-class consensus motifs without supervision. *If Neobe's promoter design tooling has only been validated on random splits of internal data, this memorization problem is almost certainly hiding in your numbers.*

**Foundation models give cheap ranking but biased calibration.** DNABERT-6 (89M-param BERT pretrained on multi-species DNA) matches CNN/k-mer on the easy splits and *exceeds* both on Spearman ranking on leave-spacer-out. But on leave-m10-out it collapses to R²=0.12 while keeping Spearman at 0.73 — it ranks promoters correctly but predicts the wrong absolute log-expression by a constant offset. The pretrained priors do not perfectly transfer to E. coli σ70 context, and a regression-head fine-tune cannot fully correct the bias on ~9k examples. **For ranking-mode applications ("give me the top 20 candidates"), DNABERT-6 is fine. For calibration-mode applications ("predict absolute ON/OFF dynamic range to budget IND-grade leakage"), the small purpose-built CNN is the right model.** Knowing *when* to reach for a foundation model versus a small domain model is the practical question for biotech ML adoption — and the answer at IND scale leans toward the small model trained on your own characterization data.

## What selectivity actually means at IND

Selectivity for an intratumoral live biotherapeutic is a *budget* across at least three tunable layers:

1. **Sensor specificity** — the ON/OFF transcription factor that responds to the tumor cue (FNR for hypoxia, others for low pH, lactate, kynurenine). Sets the floor on leakage.
2. **Promoter strength under condition** — converts sensor signal into payload mRNA. Tunable across orders of magnitude.
3. **RBS / translation initiation** — converts mRNA into protein. Independent knob, often more tunable than the promoter once the promoter is fixed.

The IND-relevant safety story is **fold-change between tumor-conditioned and healthy-tissue-conditioned expression**, not absolute strength. The Halozyme / PEGPH20 cautionary tale is the systemic-delivery analog of this: a payload that worked in vitro failed at Ph3 because off-target enzyme activity in the bloodstream produced toxicity. Your bacterial localization + circuit-driven expression is the architectural answer to that failure mode, but the engineering team still has to *quantitatively budget* the leak across the three layers above.

## Where I'd contribute in the first 90 days

**Month 1 — characterize the budget.**
Audit your existing in-house promoter / sensor / RBS characterization data. Build a per-component characterization model so we have a quantitative prior on what each part contributes to total leakage and total dynamic range. If internal MPRA-quality data exists at ≥1000 variants, fine-tune PromoterNet on it and we have a proprietary in-distribution predictor; if not, identify the cheapest experiment that would give us one. This is the data prerequisite for everything downstream.

**Month 2 — leak budgeter for the existing lead circuit.**
Take the chassis + sensor + promoter + RBS combination going into the lead candidate and produce a per-organ predicted leakage table (gut, liver, spleen, tumor) under literature-grounded condition assumptions. If ranges are wide, recommend the single experiment that closes the largest uncertainty band. The goal is to give the wet-lab team a one-page predicted-leak target before they start round-N variants.

**Month 3 — in-silico promoter library proposal.**
Generate a ranked candidate list of new hypoxia/lactate-responsive promoter variants with target dynamic-range characteristics, with each candidate scored by predicted ON, predicted OFF, predicted dynamic range, and a calibrated uncertainty. Hand off the top 20 for synthesis and characterization. The CV on the predictions tells you which candidates are real bets vs. extrapolations the model is bluffing on.

## Three things I am explicitly *not* asking to own

1. **Payload protein engineering.** You have that direction covered, and the wins are smaller relative to the selectivity gains.
2. **Kill-switch architecture.** Your co-founder's domain. I'd want to read what's been standardized and stay out of the way.
3. **Chassis selection.** This is a wet-biology decision that should be informed by colonization data the in-vivo team owns, not by sequence models.

## Why this is differentiated relative to "use a transformer for it"

Most computational pitches a synbio company hears are flavor-of-the-month tooling without an evaluation story tied to deployment. The line above — **PromoterNet survives leave-element-out at R²=0.67 where k-mer collapses to R²=0.23** — is the kind of evaluation a CRO can't fake and an investor recognizes. It's also a discipline transferable to your sensor design problem: any model trained on "old sensors" will have the same memorization failure mode against "new sensor variants." Setting up the right held-out evaluation upfront is the cheap insurance.

## Asks for the next conversation

1. **What's the most informative existing internal dataset I could be cleared to look at on a short consulting agreement?** (Even a paper-equivalent NDA scope is fine.)
2. **What dynamic range does the lead circuit need to hit for IND, and how much of it currently comes from sensor vs promoter vs RBS?**
3. **Are there metabolite cues in the tumor microenvironment you wish you had a sensor for but no natural bacterial TF responds to?** That's the place where my protein-design stack (RFdiffusion / ProteinMPNN / AlphaFold) lands inside the selectivity layer rather than the payload layer — a sensor-domain redesign sprint.

---

Repo: `C:/Users/Kirill/promoternet/`. Happy to walk through any of the figures or training runs live.
