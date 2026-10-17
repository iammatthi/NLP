# Cross-Lingual Depression Severity Classification

Research project on detecting **mental health severity in low-resource languages**, comparing
cross-lingual transfer strategies against native fine-tuning and zero-/few-shot LLMs.

## Research questions

**RQ1.** How effective are cross-lingual transfer strategies compared with native fine-tuning
and zero-/few-shot LLMs for detecting mental health severity in low-resource languages?

**RQ2 (exploratory).** How do translation artifacts distort token-level feature attributions
(SHAP) and contribute to clinical severity downgrades?

## Experimental configurations

| # | Condition | Train data | Backbone | Test data |
|---|-----------|------------|----------|-----------|
| 1 | Zero-Shot Cross-Lingual Transfer | EN-train | XLM-R | BN-test (native) |
| 2 | Translate-Train | EN-train → translated to BN | **XLM-R** | BN-test (native) |
| 3 | Translate-Test | EN-train | **same XLM-R as #1** | BN-test → translated to EN |
| **4a** | Native (controlled) | BN-train | **XLM-R** | BN-test (native) |
| **4b** | Native (specialized) | BN-train | **BanglaBERT** | BN-test (native) |
| 5 | LLM baselines (zero-/few-shot) | prompts | proprietary + compact open-weight | BN-test (native) |

Design rationale:

- **#1 vs #3** share identical model weights — only the *test representation* differs. This is the
  cleanest pairing in the paper and directly feeds RQ2.
- **#1/#2/#4a** share an identical backbone — isolates the *training-data strategy* (raw EN vs.
  translated EN vs. native BN).
- **#4b** is the "how much does a native LM add beyond native data" oracle. Report it in the same
  table but label it as non-controlled (architecture, tokenizer and pretraining corpus all differ).

## Datasets

Both files are byte-identical to their published sources (verified by md5).

| File | Source | Language | Labels | n | License |
|------|--------|----------|--------|---|---------|
| `data/raw/depression-severity-bengali.xlsx` | Kabir et al. 2022, *JMIR Formative Research* — [omanwhatiscomputer/depression-severity](https://github.com/omanwhatiscomputer/depression-severity) | Bengali (blogs, microblogs, Facebook groups) | 4-level ordinal int `1–4`, DSM-5-adherent, validated by BRAC University counselors | 4897 | none stated (article is CC BY 4.0) |
| `data/raw/depression-severity-english.csv` | Naseem et al. 2022, *WWW* — [usmaann/Depression_Severity_Dataset](https://github.com/usmaann/Depression_Severity_Dataset) | English (Reddit) | 4-level ordinal `minimum/mild/moderate/severe`, BDI + Depressive Disorder Annotation | 3553 | none stated |

These are the only two freely downloadable, text-bearing, 4-level ordinal depression-severity
datasets in the [bucuram/depression-datasets-nlp](https://github.com/bucuram/depression-datasets-nlp)
and [bucuram/multilingual-mental-health-datasets-nlp](https://github.com/bucuram/multilingual-mental-health-datasets-nlp)
surveys. Everything else is either binary, not severity-labeled, or gated behind a DUA.

### Unified label schema

| Level | Name | Bengali (`labels`) | English (`label`) |
|-------|------|--------------------|-------------------|
| 1 | minimal | `1` | `minimum` |
| 2 | mild | `2` | `mild` |
| 3 | moderate | `3` | `moderate` |
| 4 | severe | `4` | `severe` |

Bengali level semantics (from the Kabir et al. 2022 paper): **L4** suicidal attempts/inclinations,
self-harm, schizophrenia/BPD dx; **L3** need for counseling/medication, clinical depression with
physical symptoms; **L2** general depression, hopelessness, loneliness, low self-esteem; **L1**
least problematic, occasional unhappiness.

### Known asymmetries (must be reported, not hidden)

**Label prior mismatch** — the single largest confound:

| | minimal | mild | moderate | severe |
|---|---------|------|----------|--------|
| Bengali (Kabir) | 2097 (43%) | 1540 (31%) | 707 (14%) | 553 (11%) |
| English (Naseem) | 2587 (**73%**) | 290 (8%) | 394 (11%) | 282 (8%) |

English `mild` is *smaller* than `moderate` (non-monotonic). A model trained on English learns a
73%-minimal prior and will systematically **downgrade** Bengali cases — which is exactly the
failure mode RQ2 studies. This is a *finding*, but it must be separated from the transfer-strategy
effect.

Other asymmetries: Reddit posts (median ~80 words) vs. Bengali blogs/Facebook (median ~22 words);
DSM-5 tiers vs. BDI-derived annotation; no inter-annotator kappa published for the Bengali set.

### Dataset handling decision

**Do not resample or rebalance the raw datasets.** Reasons: discards ~2,000 real English
`minimum` posts while oversampling the 290 `mild` posts (overfitting the hardest class); uses the
test-set label prior to shape training (label-shift leakage); and the test side is already constant
across all five configurations (same Bengali test items, possibly translated).

Instead, two cheap controls:

1. **Class-weighted loss** (inverse-frequency, computed from each run's own train split) in every
   run. No test information used.
2. **One logit-adjustment ablation** on configs 1–3: post-hoc subtract `τ · log p_target(y)` from
   the logits, *no retraining*. If the EN→BN gap collapses, the finding is "transfer works, priors
   don't"; if it doesn't, it is a genuine language-transfer result.

## Models

Common backbone for all controlled comparisons: **`FacebookAI/xlm-roberta-base`**
(RoBERTa, 12L/768h, 250k SentencePiece). Selected because it is the only candidate that can serve
all four controlled configurations — including zero-shot cross-lingual transfer, which requires a
shared multilingual embedding space.

`csebuetnlp/banglabert` (ELECTRA-base, 12L/768h, 32k Bangla WordPiece, pre-trained on 16.6GB of
Bangla) is used **only** for condition 4b. It is the strongest Bengali-native LM but is
architecturally incompatible with XLM-R (ELECTRA vs. RoBERTa pretraining, Bangla-only tokenizer),
so using it as the primary native condition would confound *strategy* with *model choice*.

RQ2 (SHAP) runs exclusively on XLM-R rows: token-level attributions are only comparable within a
single tokenizer, and XLM-R vs. BanglaBERT segment the same sentence differently.

## Metrics

Severity is **ordinal**, so accuracy alone is misleading and cannot express "downgrades".
Report, per configuration:

- macro-F1 and per-class F1 (primary; neutralises the prior mismatch at the metric level)
- MAE over the ordinal levels
- quadratic weighted kappa
- **upgrade / ok / downgrade error-direction matrix** (downgrade = `pred < gold`) — the core RQ2 measure
- bootstrap 95% CIs and McNemar tests between configurations

## Repository layout

```
src/schema.py          unified label mapping, stratified 80/10/10 splits, fixed seed
src/dataset.py         pandas frame -> tokenised model inputs (labels as model classes 0..3)
src/losses.py          class-weighted cross-entropy for the HF Trainer
src/metrics.py         ordinal metrics + Trainer compute_metrics hook
scripts/
  prepare_splits.py    emit data/splits/{bn,en}_{train,val,test}.csv
  smoke_test_model.py  CPU-only build check (no training)
  push_checkpoint.py   upload one run's best checkpoint to a private HF Hub repo
train.py               fine-tune a backbone on the native splits (experiment 4)
evaluate.py            ordinal metrics + bootstrap CIs on a held-out split
notebooks/
  experiment4_native.ipynb   Colab driver for conditions 4a and 4b
data/
  raw/                 source datasets (do not name this `datasets/` - it shadows the HF package)
  splits/              generated stratified splits (gitignored, regenerated by seed)
outputs/               run artefacts: checkpoints, metrics, predictions (gitignored)
```

Labels are stored in domain space (1..4) everywhere except inside the model
boundary, where `to_model_label` / `to_domain_label` shift to 0..3 and back.

## Running it

[`notebooks/experiment4_native.ipynb`](notebooks/experiment4_native.ipynb) drives
experiment 4: it sets up Colab, trains 4a and 4b across 3 seeds each, scores the
held-out test split, and pushes every best checkpoint to a private Hub repo as it
finishes. Run the notebook end to end.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iammatthi/NLP/blob/main/notebooks/experiment4_native.ipynb)

Entry points — see each `--help` for flags, and the notebook for the invocation
that ties them together:

- `scripts/prepare_splits.py` — stratified 80/10/10 splits, fixed seed
- `scripts/smoke_test_model.py` — CPU build check for both backbones
- `train.py` — fine-tune one backbone on the native splits
- `evaluate.py` — ordinal metrics + bootstrap CIs on a held-out split
- `scripts/push_checkpoint.py` — upload one run's best checkpoint to the Hub

Conventions the notebook assumes:

- **3 seeds per condition**, reported as `mean ± std`.
- Checkpoints are selected on **validation macro-F1**; `evaluate.py` then scores
  the separate held-out **test** split.
- Class-weighted cross-entropy (inverse-frequency from the train split) is on by
  default — without it the model collapses onto `minimal`.
- Checkpoints are pushed to a **private** Hub repo (`--repo-id` is required, the
  tool has no default). Keep it private: the source datasets state no licence, so
  trained weights should not be published until that is settled.

## Implementation status

| Step | Status |
|------|--------|
| Schema + stratified splits | done (`src/schema.py`, `scripts/prepare_splits.py`) |
| Local CPU build check | done (`scripts/smoke_test_model.py`) |
| Training loop, class-weighted CE, early stopping | done (`train.py`) |
| Ordinal metrics + bootstrap CIs | done (`evaluate.py`) |
| Experiment 4a / 4b end-to-end | verified on CPU with `--max-steps 4` |
| Colab notebook for experiment 4 | done (`notebooks/experiment4_native.ipynb`) |
| Checkpoint persistence (private HF Hub repo) | done (`scripts/push_checkpoint.py`) |
| Translation (configs 2–3) | not started |
| Logit-adjustment ablation | not started |
| SHAP (RQ2) | not started |
| LLM baselines (config 5) | not started |

## Out of scope

Dropped to keep the timeline short; revisit only if time permits.

- HelaDepDet 2023 (English, 40K, 4-level) verification as a larger English severity set
- DEPTWEET DUA application (scientifically closest English counterpart — same DSM-5/PHQ-9-inspired
  typology as the Bengali set — but IDs only, DUA-gated, and has documented duplicate-ID label defects)
- Roman Urdu (Mohmand et al. 2024, 25K, 4-level, free) as a second low-resource language
- Any backbone A/B beyond XLM-R vs. BanglaBERT on configuration 4
- Extensive hyperparameter sweeps

## Getting started

Local development only — the Colab notebook uses `pip` instead.

```bash
uv sync
```

Then see **Running it**.
