# Experiment 2 — Translate-Train (English → Bengali)

Condition 2 of the research design: fine-tune XLM-R on English posts **translated to Bengali**, then evaluate on the **native Bengali** test split. This isolates the effect of replacing raw English supervision with machine-translated Bengali supervision, against the zero-shot cross-lingual transfer baseline (Condition 1) and the native Bengali gold standard (Condition 4a).

|            | 2 (translate-train)                          | 1 (zero-shot, 10-ep tuned)                | 4a (native controlled)                    |
| ---------- | -------------------------------------------- | ----------------------------------------- | ----------------------------------------- |
| backbone   | `FacebookAI/xlm-roberta-base`                | `FacebookAI/xlm-roberta-base`             | `FacebookAI/xlm-roberta-base`             |
| train / val| `en-train` → bn (n=2842) / `en-val` → bn (n=355) | `en-train` (n=2842) / `en-val` (n=355) | `bn-train` (n=3917) / `bn-val` (n=490) |
| test       | `bn-test` (n=490, native Bengali)            | `bn-test` (n=490, native Bengali)         | `bn-test` (n=490, native Bengali)         |
| parameters | ~278M                                        | ~278M                                     | ~278M                                     |

Setup: 3 seeds (42/43/44), class-weighted cross-entropy (computed from the translated English train split), max length 256, lr 2e-5, batch 16, weight decay 0.01, up to 10 epochs, warmup 100 steps, early-stopping patience 4 — matching the tuned 10-epoch schedule of Condition 1. Best checkpoint selected on **translated English validation macro-F1**, then scored once on the held-out **native Bengali test** split (n=490). All numbers below are mean +/- std across the 3 seeds.

## Translation

English posts were translated to Bengali using Amazon Translate (standard `TranslateText` API, region `us-east-1`). The complete English dataset (3,553 rows, 1,590,489 characters) was translated once and reused across all seeds and splits. Source rows, labels and SHA-256 hashes are preserved in the translated file (`data/translated/en_to_bn.csv`) and verified before split construction. Four source rows containing only `#NAME?` were returned unchanged.

Known translation quality issues (spot-checked): negation reversal ("hopeless" → "হতাশাহীন"), literal mistranslation of idioms ("sick of" → "অসুস্থ"), added content (drug reference in row 111), and changed event semantics (row 1100). These artifacts are expected to affect model behaviour and are discussed below.

## Artefacts

Each `2_xlmr_seed{N}/` directory holds the two artefacts written by `train.py` and `evaluate.py`:

- `eval_test.json` — full ordinal metric suite on `bn-test`: macro-F1, per-class F1, MAE, QWK, accuracy, the upgrade/ok/downgrade direction matrix, the confusion matrix, the classification report, and bootstrap 95% CIs for macro-F1, QWK and the downgrade rate.
- `train_metrics.json` — training summary: final train loss, runtime, label counts and the inverse-frequency class weights.

Model checkpoints are stored in a private Hugging Face Hub repository (`iammatthi/nlp_experiment2`) for downstream reuse. The source datasets state no licence, so trained weights are not redistributed publicly.

Driver: [`notebooks/experiment2_translate_train.ipynb`](../../notebooks/experiment2_translate_train.ipynb).

## Results

| metric   | 2 Translate-Train   | 1 Zero-Shot (10-ep) | 4a Native XLM-R   |
| -------- | ------------------- | ------------------- | ----------------- |
| macro-F1 | 0.3147 +/- 0.0198   | 0.3571 +/- 0.0142   | 0.7981 +/- 0.0107 |
| QWK      | 0.1262 +/- 0.0147   | 0.2637 +/- 0.0161   | 0.8167 +/- 0.0124 |
| MAE      | 1.1109 +/- 0.0301   | 0.9367 +/- 0.0321   | 0.2327 +/- 0.0127 |
| accuracy | 0.3265 +/- 0.0221   | 0.4272 +/- 0.0031   | 0.8272 +/- 0.0085 |

| error direction                        | 2 Translate-Train   | 1 Zero-Shot (10-ep) | 4a Native XLM-R   |
| -------------------------------------- | ------------------- | ------------------- | ----------------- |
| upgrade rate (pred > gold)             | 0.4401 +/- 0.0145   | 0.2381 +/- 0.0165   | 0.1000 +/- 0.0201 |
| downgrade rate (pred < gold)           | 0.2333 +/- 0.0205   | 0.3347 +/- 0.0184   | 0.0728 +/- 0.0136 |
| error rate                             | 0.6735 +/- 0.0221   | 0.5728 +/- 0.0031   | 0.1728 +/- 0.0085 |
| downgrade share of errors (down / err) | 0.3462 +/- 0.0228   | 0.5843 +/- 0.0301   | 0.4236 +/- 0.0920 |

|                                 | 2 Translate-Train   | 1 Zero-Shot (10-ep) | 4a Native XLM-R   |
| ------------------------------- | ------------------- | ------------------- | ----------------- |
| mean signed error (pred - gold) | 0.4660 +/- 0.0301   | -0.0184 +/- 0.0827  | 0.0299 +/- 0.0325 |

Per-class F1:

| class    | test support | 2 Translate-Train   | 1 Zero-Shot (10-ep) | 4a Native XLM-R   |
| -------- | ------------ | ------------------- | ------------------- | ----------------- |
| minimal  | 210          | 0.3852 +/- 0.0088   | 0.5714 +/- 0.0222   | 0.8965 +/- 0.0052 |
| mild     | 154          | 0.3018 +/- 0.0673   | 0.1242 +/- 0.0856   | 0.8269 +/- 0.0144 |
| moderate | 71           | 0.2972 +/- 0.0151   | 0.3727 +/- 0.0063   | 0.6203 +/- 0.0128 |
| severe   | 55           | 0.2745 +/- 0.0265   | 0.3603 +/- 0.0337   | 0.8488 +/- 0.0346 |

Precision / recall (mean across seeds):

| class    | 2 Translate-Train P / R | 1 Zero-Shot (10-ep) P / R | 4a Native XLM-R P / R |
| -------- | ----------------------- | ------------------------- | --------------------- |
| minimal  | 0.4916 / 0.3175         | 0.4897 / 0.6857           | 0.9133 / 0.8810       |
| mild     | 0.3217 / 0.2857         | 0.3488 / 0.0779           | 0.8269 / 0.8268       |
| moderate | 0.2523 / 0.3615         | 0.3680 / 0.3803           | 0.5950 / 0.6526       |
| severe   | 0.2017 / 0.4303         | 0.2963 / 0.4788           | 0.8498 / 0.8485       |

Class weights used by the loss (computed on the translated English train split):

| class    | EN train count | class weight |
| -------- | -------------- | ------------ |
| minimal  | 2069 (72.8%)   | 0.1560       |
| mild     | 232 (8.2%)     | 1.3912       |
| moderate | 315 (11.1%)    | 1.0246       |
| severe   | 226 (8.0%)     | 1.4281       |

Validation checkpoint selection on the translated English validation split (n=355), from the training logs. All three seeds completed the full 10-epoch schedule; none triggered early stopping.

| seed | best val macro-F1 | best epoch |
| ---- | ----------------- | ---------- |
| 42   | 0.4836            | 10         |
| 43   | 0.5103            | 6          |
| 44   | 0.5464            | 7          |

In all seeds, validation loss reached its minimum around epochs 2-4 (1.186-1.211) and then rose monotonically to 1.76-1.93 by epoch 10, while macro-F1 kept improving or held steady. Checkpoint selection therefore never coincided with the loss minimum.

## Interpretation

### Translate-train is worse than zero-shot transfer

The headline finding is negative: **translating the English training data to Bengali produces a model that is worse than simply training on raw English and transferring zero-shot.** Macro-F1 drops from 0.3571 (zero-shot) to 0.3147 (translate-train), QWK from 0.2637 to 0.1262, and MAE rises from 0.9367 to 1.1109. The gap is not attributable to training schedule — both conditions use the identical 10-epoch tuned recipe.

This result is consistent with the translation quality issues identified during spot-checking. Machine translation introduces systematic artifacts — reversed negation, literal idiom mistranslation, added content, shifted event semantics — that the model learns as if they were genuine Bengali. When evaluated on native Bengali posts that lack these artifacts, the model's learned representations are misaligned. In contrast, zero-shot transfer at least benefits from clean English source text and a shared multilingual embedding space that was not distorted by translation noise.

### Error direction: strong over-estimation

Translate-train predominantly **upgrades** severity (upgrade rate 0.4401 vs. downgrade rate 0.2333; mean signed error +0.4660), the opposite of Condition 1, whose errors skew toward downgrades (downgrade share of errors 0.5843).

The mean confusion matrix shows where this upward shift occurs:

| predicted → | minimal | mild | moderate | severe |
|-------------|---------|------|----------|--------|
| **minimal** (210) | 67 | 59 | 40 | 45 |
| **mild** (154) | 45 | 44 | 24 | 41 |
| **moderate** (71) | 18 | 20 | 26 | 7 |
| **severe** (55) | 6 | 13 | 12 | 24 |

Predicted marginals: minimal 136, mild 136, moderate 102, severe 117. The model **under-predicts `minimal`** (-74 cases) and **over-predicts `moderate`** (+31) and `severe` (+62). This is the opposite of Condition 1, where `minimal` was over-predicted by +84 cases and `mild` collapsed by -123.

A plausible mechanism: translation normalises emotional intensity and collapses lexical distinctions between severity tiers. A `minimal` English post about everyday unhappiness, when translated to Bengali, may adopt vocabulary that overlaps with `moderate` or `severe` native Bengali expressions. The model learns these inflated mappings and applies them to native test posts.

### Seed variability

Test macro-F1 varies by +/- 0.0198 across seeds (0.2919 / 0.3282 / 0.3239) — comparable to Condition 1 (+/- 0.0142) and tighter than the earlier runs of this condition. All three seeds reached `severe` F1 between 0.245 and 0.297; no class collapses in any seed.

One reproducibility caveat: the **first** seed-44 attempt destabilized — its training loss stayed at the untrained level (~1.39) for five epochs after a gradient-norm spike of ~206 early in epoch 2, its validation macro-F1 flat-lined at the uniform-random baseline, and its test predictions contained no `severe` cases at all. The run was repeated with the same seed and trained normally to 10 epochs; the artefacts above are from the successful rerun. Since seed alone does not deterministically control the outcome on this stack (bf16 plus dataloader workers), the instability is best read as run-level nondeterminism rather than a property of the configuration — but it does show this condition can fail catastrophically in individual runs.

### Class-level performance

All four classes perform well below Condition 4a, and the pattern differs from Condition 1:

| class    | 2 Translate-Train | 1 Zero-Shot (10-ep) | 4a Native |
| -------- | ----------------- | ------------------- | --------- |
| minimal  | 0.3852            | 0.5714              | 0.8965    |
| mild     | 0.3018            | 0.1242              | 0.8269    |
| moderate | 0.2972            | 0.3727              | 0.6203    |
| severe   | 0.2745            | 0.3603              | 0.8488    |

`mild` improves relative to zero-shot (0.3018 vs. 0.1242) — translation may help the model see Bengali surface forms for the interior class. `minimal` degrades sharply (0.3852 vs. 0.5714): translated `minimal` posts lose the lexical markers that let the zero-shot model recognise the floor class, and the model pushes those cases upward instead. `severe` recall is high (0.4303) but precision is the lowest of any class (0.2017), consistent with the general upward shift.

### Training dynamics: loss–F1 divergence

Validation loss diverges from macro-F1 after epochs 2-4 in all seeds (e.g. seed 42: loss 1.211 at epoch 4 → 1.846 at epoch 10, while macro-F1 rises 0.4179 → 0.4836). This is expected: cross-entropy penalises miscalibrated probabilities, while macro-F1 only cares about the argmax decision. The model becomes increasingly overconfident in its (often wrong) predictions, which inflates loss without changing the top-1 class. Because checkpoint selection uses macro-F1, the selected checkpoints (epochs 10, 6, 7) capture the best decision quality rather than the best-calibrated probabilities; downstream uses that consume probabilities rather than argmax decisions would need recalibration or earlier checkpoints.

## Limitations

- **n=3 seeds.** The +/- values are sample standard deviations over three runs and describe seed spread, not confidence intervals over test examples.
- Per-run **bootstrap 95% CIs** for macro-F1, QWK and the downgrade rate are stored inside each `eval_test.json` under `ci95`.
- **Translation quality is not formally evaluated.** Spot-checks identified specific errors (negation reversal, idiom mistranslation, added content) but no systematic human evaluation was conducted. The extent to which translation artifacts explain the performance gap is inferred, not measured.
- **Single translation system.** Amazon Translate is the only provider used. Other systems (IndicTrans2, Google NMT, Azure Translator) may produce different artifact profiles.
- **Run-level nondeterminism.** As documented above, one run of this condition destabilized and had to be repeated. The reported aggregates come from successful runs; the failure rate of this configuration under repeated sampling is unknown.
- **Confounded comparison with Condition 1.** The two conditions differ in training data language (translated BN vs. raw EN), which simultaneously changes the surface language, the lexical distribution, and the translation artifact profile. Separating these effects would require an ablation (e.g., training on back-translated English).
- The test split is a single held-out set of 490 examples; per-class estimates, especially for `severe` (55 cases), have limited statistical power.
- The source datasets state no licence. Trained weights should not be redistributed until that is settled.
