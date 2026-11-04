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
| macro-F1 | 0.2850 +/- 0.0472   | 0.3571 +/- 0.0142   | 0.7682 +/- 0.0313 |
| QWK      | 0.1419 +/- 0.0403   | 0.2637 +/- 0.0161   | 0.7798 +/- 0.0136 |
| MAE      | 1.0463 +/- 0.1247   | 0.9367 +/- 0.0321   | 0.2707 +/- 0.0290 |
| accuracy | 0.3299 +/- 0.0242   | 0.4272 +/- 0.0031   | 0.8054 +/- 0.0286 |

| error direction                        | 2 Translate-Train   | 1 Zero-Shot (10-ep) | 4a Native XLM-R   |
| -------------------------------------- | ------------------- | ------------------- | ----------------- |
| upgrade rate (pred > gold)             | 0.3952 +/- 0.0672   | 0.2381 +/- 0.0165   | 0.1095 +/- 0.0125 |
| downgrade rate (pred < gold)           | 0.2748 +/- 0.0567   | 0.3347 +/- 0.0184   | 0.0850 +/- 0.0183 |
| error rate                             | 0.6701 +/- 0.0242   | 0.5728 +/- 0.0031   | 0.1946 +/- 0.0286 |
| downgrade share of errors (down / err) | 0.4110 +/- 0.0908   | 0.5843 +/- 0.0301   | 0.4348 +/- 0.0368 |

|                                 | 2 Translate-Train   | 1 Zero-Shot (10-ep) | 4a Native XLM-R   |
| ------------------------------- | ------------------- | ------------------- | ----------------- |
| mean signed error (pred - gold) | 0.2694 +/- 0.3111   | -0.0184 +/- 0.0827  | 0.0381 +/- 0.0139 |

Per-class F1:

| class    | test support | 2 Translate-Train   | 1 Zero-Shot (10-ep) | 4a Native XLM-R   |
| -------- | ------------ | ------------------- | ------------------- | ----------------- |
| minimal  | 210          | 0.4391 +/- 0.0953   | 0.5714 +/- 0.0222   | 0.8790 +/- 0.0169 |
| mild     | 154          | 0.2584 +/- 0.0950   | 0.1242 +/- 0.0856   | 0.8143 +/- 0.0390 |
| moderate | 71           | 0.2669 +/- 0.0518   | 0.3727 +/- 0.0063   | 0.5950 +/- 0.0400 |
| severe   | 55           | 0.1754 +/- 0.1530   | 0.3603 +/- 0.0337   | 0.7845 +/- 0.0311 |

Precision / recall (mean across seeds):

| class    | 2 Translate-Train P / R | 1 Zero-Shot (10-ep) P / R | 4a Native XLM-R P / R |
| -------- | ----------------------- | ------------------------- | --------------------- |
| minimal  | 0.4942 / 0.4063         | 0.4897 / 0.6857           | 0.8889 / 0.8698       |
| mild     | 0.2896 / 0.2359         | 0.3488 / 0.0779           | 0.8305 / 0.8009       |
| moderate | 0.2180 / 0.3521         | 0.3680 / 0.3803           | 0.5776 / 0.6150       |
| severe   | 0.1295 / 0.2727         | 0.2963 / 0.4788           | 0.7538 / 0.8182       |

Class weights used by the loss (computed on the translated English train split):

| class    | EN train count | class weight |
| -------- | -------------- | ------------ |
| minimal  | 2069 (72.8%)   | 0.1560       |
| mild     | 232 (8.2%)     | 1.3912       |
| moderate | 315 (11.1%)    | 1.0246       |
| severe   | 226 (8.0%)     | 1.4281       |

Validation trajectory on translated English (`en-val` → bn, n=355) during training (seed 42; seeds 42 and 43 completed 10 epochs, seed 44 triggered early stopping at epoch 5):

| epoch | val macro-F1 | val loss | val MAE |
| ----- | ------------ | -------- | ------- |
| 1     | 0.2089       | 1.353    | 1.045   |
| 2     | 0.2326       | 1.342    | 1.180   |
| 3     | 0.4085       | 1.222    | 0.780   |
| 4     | 0.4421       | 1.291    | 0.600   |
| 5     | 0.4488       | 1.354    | 0.609   |
| 6     | 0.4526       | 1.313    | 0.670   |
| 7     | 0.4569       | 1.443    | 0.716   |
| 8     | 0.4720       | 1.620    | 0.617   |
| 9*    | 0.4816       | 1.697    | 0.580   |
| 10*   | 0.4804       | 1.760    | 0.606   |

*\*Seed 42 selected epoch 9 as the best checkpoint. Seed 44 triggered early stopping at epoch 5. Seeds 42 and 43 trained the full 10 epochs because macro-F1 improved marginally each epoch; validation loss diverged after epoch 3 in all seeds.*

## Interpretation

### Translate-train is worse than zero-shot transfer

The headline finding is negative: **translating the English training data to Bengali produces a model that is worse than simply training on raw English and transferring zero-shot.** Macro-F1 drops from 0.3571 (zero-shot) to 0.2850 (translate-train), QWK from 0.2637 to 0.1419, and MAE rises from 0.9367 to 1.0463. The gap is not attributable to training schedule — both conditions use the identical 10-epoch tuned recipe.

This result is consistent with the translation quality issues identified during spot-checking. Machine translation introduces systematic artifacts — reversed negation, literal idiom mistranslation, added content, shifted event semantics — that the model learns as if they were genuine Bengali. When evaluated on native Bengali posts that lack these artifacts, the model's learned representations are misaligned. In contrast, zero-shot transfer at least benefits from clean English source text and a shared multilingual embedding space that was not distorted by translation noise.

### Error direction flips: over-estimation instead of under-estimation

The most striking qualitative difference from Condition 1 is the **direction of errors**. Zero-shot transfer predominantly **downgrades** severity (downgrade share of errors 0.5843), consistent with the English `minimal`-heavy prior pulling predictions toward the floor. Translate-train predominantly **upgrades** severity (upgrade rate 0.3952 vs. downgrade rate 0.2748; mean signed error +0.2694).

The mean confusion matrix shows where this upward shift occurs:

| predicted → | minimal | mild | moderate | severe |
|-------------|---------|------|----------|--------|
| **minimal** (210) | 85 | 53 | 41 | 29 |
| **mild** (154) | 53 | 36 | 37 | 28 |
| **moderate** (71) | 23 | 18 | 25 | 4 |
| **severe** (55) | 10 | 13 | 17 | 15 |

Predicted marginals: minimal 171, mild 121, moderate 120, severe 77. The model **under-predicts `minimal`** (-39 cases) and **over-predicts `moderate`** (+49 cases) and `severe` (+22 cases). This is the opposite of Condition 1, where `minimal` was over-predicted by +84 cases and `mild` collapsed by -123.

A plausible mechanism: translation normalises emotional intensity and collapses lexical distinctions between severity tiers. A `minimal` English post about everyday unhappiness, when translated to Bengali, may adopt vocabulary that overlaps with `moderate` or `severe` native Bengali expressions. The model learns these inflated mappings and applies them to native test posts.

### Seed variability is high

Test macro-F1 varies by +/- 0.0472 across seeds — substantially more than Condition 1 (+/- 0.0142) or Condition 4a (+/- 0.0313). Seed 44 is an outlier: it triggered early stopping at epoch 5 and produced `severe` F1 = 0.000 (the model never predicted `severe` on any test case), while seeds 42 and 43 reached `severe` F1 of 0.281 and 0.245.

The confusion matrix for seed 44 shows the collapse explicitly: the `severe` column is entirely zero across all true classes. This is consistent with early stopping preventing the model from learning the `severe` decision boundary — the class has only 226 training examples (8.0%) with weight 1.4281, and the translated examples may not carry the strong lexical cues (self-harm, suicidal ideation) that distinguish `severe` in native Bengali.

### Class-level performance

All four classes perform well below Condition 4a, and the pattern differs from Condition 1:

| class    | 2 Translate-Train | 1 Zero-Shot (10-ep) | 4a Native |
| -------- | ----------------- | ------------------- | --------- |
| minimal  | 0.4391            | 0.5714              | 0.8790    |
| mild     | 0.2584            | 0.1242              | 0.8143    |
| moderate | 0.2669            | 0.3727              | 0.5950    |
| severe   | 0.1754            | 0.3603              | 0.7845    |

`mild` improves relative to zero-shot (0.2584 vs. 0.1242) — translation may help the model see Bengali surface forms for the interior class. But `severe` degrades sharply (0.1754 vs. 0.3603), driven by seed 44's complete collapse. `moderate` also drops (0.2669 vs. 0.3727). The net effect is a worse overall macro-F1.

### Training dynamics: loss–F1 divergence

Validation loss diverges from macro-F1 after epoch 3 in all seeds (seed 42: loss 1.222 at epoch 3 → 1.760 at epoch 10, while macro-F1 rises 0.4085 → 0.4816). This is expected: cross-entropy penalises miscalibrated probabilities, while macro-F1 only cares about the argmax decision. The model becomes increasingly overconfident in its (often wrong) predictions, which inflates loss without changing the top-1 class. Because checkpoint selection uses macro-F1, the best model (epoch 9 for seed 42) is captured before the loss divergence becomes extreme.

## Limitations

- **n=3 seeds.** The +/- values are sample standard deviations over three runs and describe seed spread, not confidence intervals over test examples.
- Per-run **bootstrap 95% CIs** for macro-F1, QWK and the downgrade rate are stored inside each `eval_test.json` under `ci95`.
- **Translation quality is not formally evaluated.** Spot-checks identified specific errors (negation reversal, idiom mistranslation, added content) but no systematic human evaluation was conducted. The extent to which translation artifacts explain the performance gap is inferred, not measured.
- **Single translation system.** Amazon Translate is the only provider used. Other systems (IndicTrans2, Google NMT, Azure Translator) may produce different artifact profiles.
- **Seed 44 anomaly.** Early stopping at epoch 5 and complete `severe` collapse inflates the seed spread. With 3 seeds, one outlier significantly affects the mean and std.
- **Confounded comparison with Condition 1.** The two conditions differ in training data language (translated BN vs. raw EN), which simultaneously changes the surface language, the lexical distribution, and the translation artifact profile. Separating these effects would require an ablation (e.g., training on back-translated English).
- The test split is a single held-out set of 490 examples; per-class estimates, especially for `severe` (55 cases), have limited statistical power.
- The source datasets state no licence. Trained weights should not be redistributed until that is settled.
