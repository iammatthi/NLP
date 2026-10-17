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
train.py               fine-tune a backbone on the native splits (experiment 4)
evaluate.py            ordinal metrics + bootstrap CIs on a held-out split
data/
  raw/                 source datasets (do not name this `datasets/` - it shadows the HF package)
  splits/              generated stratified splits
outputs/               run artefacts: checkpoints, metrics, predictions
```

Labels are stored in domain space (1..4) everywhere except inside the model
boundary, where `to_model_label` / `to_domain_label` shift to 0..3 and back.

## Replicating experiment 4

Experiment 4 is the **Native Gold Standard**: train and evaluate directly on
native Bengali splits. 4a uses the shared XLM-R backbone (the controlled
condition comparable to configs 1–3), 4b uses BanglaBERT (the native-specialized
oracle). Both read the *same* splits, so the only difference is the backbone.

### 0. Setup

```bash
uv sync
```

### 1. Verify the models build locally (CPU, no training)

Catches checkpoint-loading problems before spending Colab time. Exits non-zero
on failure.

```bash
uv run python scripts/smoke_test_model.py
```

### 2. Prepare the stratified splits

Written once, reused by every run, so results are comparable across seeds.

```bash
uv run python scripts/prepare_splits.py --lang bn --seed 42
uv run python scripts/prepare_splits.py --lang en --seed 42   # configs 1-3 later
```

### 3. Train

3 seeds per condition (README reports mean ± std). Checkpoints are selected on
validation macro-F1; the best checkpoint is written to `<output-dir>/final/`.

```bash
# 4a - XLM-R on native Bengali (controlled condition)
uv run python train.py --model xlmr --lang bn --output-dir outputs/4a_xlmr_seed42 \
    --seed 42 --epochs 4 --train-batch-size 16 --learning-rate 2e-5
uv run python train.py --model xlmr --lang bn --output-dir outputs/4a_xlmr_seed43 \
    --seed 43 --epochs 4 --train-batch-size 16 --learning-rate 2e-5
uv run python train.py --model xlmr --lang bn --output-dir outputs/4a_xlmr_seed44 \
    --seed 44 --epochs 4 --train-batch-size 16 --learning-rate 2e-5

# 4b - BanglaBERT on native Bengali (native-specialized oracle)
uv run python train.py --model banglabert --lang bn --output-dir outputs/4b_banglabert_seed42 \
    --seed 42 --epochs 4 --train-batch-size 16 --learning-rate 2e-5
uv run python train.py --model banglabert --lang bn --output-dir outputs/4b_banglabert_seed43 \
    --seed 43 --epochs 4 --train-batch-size 16 --learning-rate 2e-5
uv run python train.py --model banglabert --lang bn --output-dir outputs/4b_banglabert_seed44 \
    --seed 44 --epochs 4 --train-batch-size 16 --learning-rate 2e-5
```

Notes:
- `--model` accepts the alias `xlmr` or `banglabert`, or any HF model id.
- Class-weighted cross-entropy is on by default (inverse-frequency, computed from
  the train split). Disable with `--no-class-weights` for an ablation.
- On a Colab GPU add `--bf16`; the script auto-disables fp16/bf16 on CPU.
- On the local CPU, verify the loop first with `--max-steps 4 --epochs 1
  --train-batch-size 4 --eval-batch-size 8` before launching a full run.

### 4. Evaluate on the held-out test split

```bash
# 4a
uv run python evaluate.py --model-dir outputs/4a_xlmr_seed42/final    --lang bn --split test --seed 42
uv run python evaluate.py --model-dir outputs/4a_xlmr_seed43/final    --lang bn --split test --seed 42
uv run python evaluate.py --model-dir outputs/4a_xlmr_seed44/final    --lang bn --split test --seed 42

# 4b
uv run python evaluate.py --model-dir outputs/4b_banglabert_seed42/final --lang bn --split test --seed 42
uv run python evaluate.py --model-dir outputs/4b_banglabert_seed43/final --lang bn --split test --seed 42
uv run python evaluate.py --model-dir outputs/4b_banglabert_seed44/final --lang bn --split test --seed 42
```

Each writes `eval_test.json` into the model directory containing macro-F1,
per-class F1, MAE, QWK, accuracy, the confusion matrix, the
upgrade/ok/downgrade error-direction matrix, and bootstrap 95% CIs.

### 5. Summarise across seeds

```bash
uv run python - <<'PY'
import json, glob, statistics as st

def mean_std(vals):
    m = st.mean(vals)
    s = st.stdev(vals) if len(vals) > 1 else 0.0
    return f"{m:.4f} ± {s:.4f}"

for tag in ("4a_xlmr", "4b_banglabert"):
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"outputs/{tag}_seed*/final/eval_test.json"))]
    if not runs:
        print(f"{tag}: no results"); continue
    for key in ("macro_f1", "qwk", "mae"):
        print(f"{tag:16s} {key:9s} {mean_std([r[key] for r in runs])}")
    print(f"{tag:16s} downgrade {mean_std([r['direction']['downgrade_rate'] for r in runs])}")
PY
```

Note `evaluate.py` writes `eval_test.json` into the `--model-dir`, which is the
`final/` subdirectory — hence the `*/final/` in the glob.

### Running on Colab

Colab has no `uv`; use the plain interpreter.

```python
# cell 1 - environment
!pip install -q "transformers==4.57.6" accelerate openpyxl pandas scikit-learn

# cell 2 - fetch the repo
!git clone <your-repo-url> repo
%cd repo
!python scripts/prepare_splits.py --lang bn --seed 42

# cell 3 - train (repeat per seed / per backbone)
!python train.py --model xlmr --lang bn --output-dir outputs/4a_xlmr_seed42 \
    --seed 42 --epochs 4 --train-batch-size 16 --learning-rate 2e-5 --bf16

# cell 4 - evaluate
!python evaluate.py --model-dir outputs/4a_xlmr_seed42/final --lang bn --split test
```

Repeat the last two lines per seed and per `--model banglabert` for 4b.

## Implementation status

| Step | Status |
|------|--------|
| Schema + stratified splits | done (`src/schema.py`, `scripts/prepare_splits.py`) |
| Local CPU build check | done (`scripts/smoke_test_model.py`) |
| Training loop, class-weighted CE, early stopping | done (`train.py`) |
| Ordinal metrics + bootstrap CIs | done (`evaluate.py`) |
| Experiment 4a / 4b end-to-end | verified on CPU with `--max-steps 4` |
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

```bash
uv sync
uv run python scripts/smoke_test_model.py
```

Then follow **Replicating experiment 4** above.
