# Experiment 3 — Translate-Test (Bengali → English)

Condition 3 of the research design: score the condition 1 checkpoints **unchanged** on Bengali
test posts **translated to English**. The same 490 held-out test items are scored twice per
checkpoint — once in their native Bengali form and once in English translation — so the two
columns below differ **only** in the test representation: the cleanest pairing in the paper and
the direct input to RQ2.

|            | 3 (translate-test)                            | paired reference (same checkpoints)           |
| ---------- | --------------------------------------------- | --------------------------------------------- |
| backbone   | `FacebookAI/xlm-roberta-base`                 | `FacebookAI/xlm-roberta-base` (identical weights) |
| train / val (for the checkpoints) | `en-train` (n=2842) / `en-val` (n=355) | identical                                |
| test       | `bn-test` → en (n=490, Amazon Translate)      | `bn-test` (n=490, native Bengali)             |
| parameters | ~278M                                         | ~278M                                         |

Setup: 3 checkpoints (training seeds 42/43/44), trained with class-weighted cross-entropy,
max length 256, lr 2e-5, batch 16, weight decay 0.01, up to 10 epochs, warmup 100 steps,
early-stopping patience 4, best checkpoint selected on English validation macro-F1. This
experiment performs **no training**: each checkpoint is scored once per representation on the
held-out test split (n=490). All numbers below are mean +/- std across the 3 checkpoints.

## Translation

Bengali posts were translated to English using Amazon Translate (standard `TranslateText` API,
region `us-east-1`). The complete Bengali dataset (4,897 rows, 1,061,931 characters) was
translated once and reused across all checkpoints and splits. Source rows, labels and SHA-256
hashes are preserved in the translated file (`data/translated/bn_to_en.csv`) and verified before
split construction; the translated test split is row-aligned with the native one. No test row
was returned with Bengali script remnants (7 rows dataset-wide retain partial Bengali tokens,
none of them in the test split).

Known translation quality issues (spot-checked): literal mistranslation that turns "trying to
get through it" into *"I tried to cut a lot"* (source row 3143) — an accidental self-harm cue;
a person shift that turns first-person *"sometimes I feel like dying"* into *"Sometimes he wants
to die"* (row 4776); an idiom direction reversal (*"I feel very envious"* → *"very jealous of
me"*, row 4583); and literal phrasing with dropped or duplicated content (row 2145: "neck"
→ "ears", "feel unwell" → *"It takes sickness"*). Overall the translations are fluent and keep
the first-person framing and the explicit distress cues ("Help me", "I don't want to die").

## Artefacts

Each `3_xlmr_seed{N}/` directory holds the four artefacts of the paired evaluation:

- `eval_test_bn2en.json` — full ordinal metric suite on `bn-test` → en: macro-F1, per-class F1,
  MAE, QWK, accuracy, the upgrade/ok/downgrade direction matrix, the confusion matrix, the
  classification report, and bootstrap 95% CIs for macro-F1, QWK and the downgrade rate.
- `eval_test_bn_native.json` — the same suite on native `bn-test` for the paired reference.
- `predictions_bn_test_en.csv` / `predictions_bn_test_native.csv` — row-aligned per-item
  predictions (`split_row`, `source_sha256`, `label`, `prediction`) for the RQ2 pairing analysis.
  `source_sha256` hashes the evaluated text, so it differs between the two files by design;
  `split_row` and `label` align them.

No model weights are produced: the checkpoints are reused unchanged. Evaluation artefacts are
pushed to a private Hugging Face Hub repository (`iammatthi/nlp_experiment3`). The source
datasets state no licence, so nothing is redistributed publicly.

Driver: [`notebooks/experiment3_translate_test.ipynb`](../../notebooks/experiment3_translate_test.ipynb).

## Results

| metric   | 3 translate-test (bn-test → en) | paired reference (bn-test native) |
| -------- | ------------------------------- | --------------------------------- |
| macro-F1 | 0.4799 +/- 0.0162               | 0.3571 +/- 0.0142                 |
| QWK      | 0.2439 +/- 0.0290               | 0.2637 +/- 0.0161                 |
| MAE      | 0.8374 +/- 0.0487               | 0.9367 +/- 0.0321                 |
| accuracy | 0.5252 +/- 0.0155               | 0.4272 +/- 0.0031                 |

| error direction                        | 3 translate-test      | paired reference      |
| -------------------------------------- | --------------------- | --------------------- |
| upgrade rate (pred > gold)             | 0.3231 +/- 0.0196     | 0.2381 +/- 0.0165     |
| downgrade rate (pred < gold)           | 0.1517 +/- 0.0096     | 0.3347 +/- 0.0184     |
| error rate                             | 0.4748 +/- 0.0155     | 0.5728 +/- 0.0031     |
| downgrade share of errors (down / err) | 0.3198 +/- 0.0244     | 0.5843 +/- 0.0301     |

|                                 | 3 translate-test      | paired reference      |
| ------------------------------- | --------------------- | --------------------- |
| mean signed error (pred - gold) | +0.4238 +/- 0.0473    | -0.0184 +/- 0.0827    |

Per-class F1:

| class    | test support | 3 translate-test      | paired reference      |
| -------- | ------------ | --------------------- | --------------------- |
| minimal  | 210          | 0.5644 +/- 0.0325     | 0.5714 +/- 0.0222     |
| mild     | 154          | 0.6299 +/- 0.0120     | 0.1242 +/- 0.0856     |
| moderate | 71           | 0.3060 +/- 0.0278     | 0.3727 +/- 0.0063     |
| severe   | 55           | 0.4195 +/- 0.0533     | 0.3603 +/- 0.0337     |

Precision / recall (mean across checkpoints):

| class    | 3 translate-test P / R | paired reference P / R |
| -------- | ---------------------- | ---------------------- |
| minimal  | 0.7956 / 0.4381        | 0.4897 / 0.6857        |
| mild     | 0.5693 / 0.7056        | 0.3488 / 0.0779        |
| moderate | 0.3057 / 0.3099        | 0.3680 / 0.3803        |
| severe   | 0.3159 / 0.6303        | 0.2963 / 0.4788        |

The mean confusion matrix for the translate-test representation (mean over the 3 checkpoints,
rounded) shows where the errors move:

| predicted → | minimal | mild | moderate | severe |
|-------------|---------|------|----------|--------|
| **minimal** (210) | 92 | 35 | 34 | 49 |
| **mild** (154) | 13 | 109 | 13 | 19 |
| **moderate** (71) | 5 | 35 | 22 | 8 |
| **severe** (55) | 5 | 12 | 4 | 35 |

Predicted marginals (from unrounded counts): minimal 116, mild 191, moderate 72, severe 111.
Against gold supports of 210/154/71/55 the model **under-predicts `minimal`** (-94 cases) and
**over-predicts `mild`** (+37) and **`severe`** (+56). The paired reference shows the opposite
bias pattern: marginals minimal 294 (+84), mild 31 (-123), moderate 74 (+3), severe 91 (+36),
with the `mild` row collapsing into `minimal` (102 of 154 cases).

### Paired per-item comparison

Between the two representations, 56–61% of all test items change their predicted class:

| checkpoint | unchanged | changed | correct only native | correct only translated | correct both | correct neither |
| ---------- | --------- | ------- | ------------------- | ----------------------- | ------------ | --------------- |
| seed 42    | 216       | 274     | 83                  | 124                     | 128          | 155             |
| seed 43    | 206       | 284     | 69                  | 126                     | 140          | 155             |
| seed 44    | 190       | 300     | 86                  | 132                     | 122          | 150             |

The discordant cells are strongly asymmetric in favour of the translated representation in
every checkpoint (McNemar exact, two-sided: p = 5.3e-3, 5.4e-5, 2.2e-3). Pooled over the 1,470
checkpoint-item pairs the counts are 238 vs. 382 (p ≈ 8.0e-9), though the pooled test reuses the
same 490 items across checkpoints and should be read as descriptive only.

Prediction transitions (pooled over the 3 checkpoints; rows = native prediction, columns =
translated prediction):

| native ↓ / translated → | minimal | mild | moderate | severe |
|-------------------------|---------|------|----------|--------|
| **minimal** (882) | 314 | 368 | 62 | 138 |
| **mild** (93) | 6 | 42 | 14 | 31 |
| **moderate** (222) | 13 | 65 | 118 | 26 |
| **severe** (273) | 14 | 98 | 23 | 138 |

The single largest flow is `minimal` → `mild` (368 of the 882 native-`minimal` predictions):
items the same model calls `minimal` in Bengali are re-read as `mild` in English translation.

## Interpretation

### Translating the test side recovers performance the shared embedding space leaves behind

The headline finding is positive: with **identical weights**, moving the test posts into the
training language raises macro-F1 from 0.3571 to 0.4799 (+0.12) and accuracy from 0.4272 to
0.5252, and cuts MAE from 0.9367 to 0.8374. Nothing about the model changes between the two
columns — the gap is purely the representation mismatch between English training text and
Bengali test text. A multilingual encoder plus a classification head trained on English does
not carry its full competence across the language boundary, and plain machine translation of
the test side unlocks a substantial part of it.

### `mild` becomes predictable; `minimal` flips from over- to under-prediction

The entire `mild` class recovers: F1 0.1242 → 0.6299, recall 0.078 → 0.706. On native Bengali
the model predicts `mild` for only 31 of 490 items despite 154 gold cases — the interior class
is effectively invisible. In translation it predicts `mild` 191 times and mostly correctly
(109 diagonal in the mean confusion matrix). The `mild` decision boundary appears to be
expressible in the training language's lexical space and not to transfer to Bengali surface
forms. The cost is `minimal` recall (0.686 → 0.438): the 294-case `minimal` over-prediction of
the reference becomes a 94-case under-prediction.

### Error direction flips from systematic downgrades to systematic upgrades

On native Bengali the errors skew downward (downgrade share of errors 0.5843, mean signed error
-0.0184) — severity is under-estimated, with `mild` cases pushed to `minimal`. Under translation
the error direction inverts: downgrade rate halves (0.3347 → 0.1517), upgrade rate rises
(0.2381 → 0.3231), and mean signed error swings to +0.4238. `severe` is over-predicted more
than twofold (111 predicted vs. 55 gold) with recall 0.63 at precision 0.32. A plausible
mechanism, consistent with the spot-checked artifacts: English renderings state distress more
explicitly than the colloquial Bengali originals ("I suffer from depression", "Help me"), and
occasional mistranslations add severity cues (the accidental self-harm cue in source row 3143).
This mechanism is inferred from examples, not measured.

### QWK is the one metric that does not improve

QWK stays flat-to-lower (0.2439 vs. 0.2637) while macro-F1 and accuracy rise. The ordinal
quadratic penalty is driven by *distance-weighted* mistakes: translation converts a large block
of two-level `minimal` → `moderate`/`severe` misses into fewer-but-farther upgrades (mean signed
error +0.42), and that systematic offset cancels the accuracy gains. The per-run bootstrap CIs
for QWK overlap heavily between the two representations (e.g. seed 43: [0.1665, 0.3524]
translated vs. [0.1536, 0.3494] native), so this should be read as "no QWK gain", not "QWK
loss".

### Most predictions change with representation, and the changes are asymmetric

274–300 of 490 items (56–61%) change predicted class when only the test representation changes,
and the changes favour the translated side in every checkpoint (McNemar exact p < 0.01 for all
three). The dominant flow `minimal` → `mild` (368 pooled) shows that translation mainly
*interiorises* predictions — it pulls items out of the `minimal` floor and into the middle of
the scale.

### Checkpoint variability is low

Macro-F1 spread is small on both representations (+/- 0.0162 translated, +/- 0.0142 native) and
the `mild` recovery is highly consistent (+/- 0.0120). Since no training happens in this
experiment, the spread reflects only differences between the three training checkpoints — the
paired design removes training variability from the comparison between representations.

## Limitations

- **n=3 checkpoints.** The +/- values are sample standard deviations over the three reused
  checkpoints and describe checkpoint spread, not confidence intervals over test examples.
  Per-run **bootstrap 95% CIs** for macro-F1, QWK and the downgrade rate are stored inside each
  `eval_test.json` under `ci95`.
- **Pooled paired statistics reuse items.** The 1,470 checkpoint-item pairs contain three
  predictions per test item; the pooled McNemar value assumes independence and is reported as
  descriptive. Per-checkpoint tests (490 independent items each) are primary and all agree.
- **Translation quality is not formally evaluated.** Spot-checks identified specific errors
  (literal self-harm cue, person shift, idiom reversal, dropped content) but no systematic human
  evaluation was conducted; how far translation artifacts explain the severity over-estimation
  is inferred, not measured.
- **Single translation system.** Amazon Translate is the only provider used. Other systems
  (IndicTrans2, Google NMT, Azure Translator) may produce different artifact profiles and
  different shifts in error direction.
- **Cross-experiment comparison is deliberately deferred.** The paired reference column is the
  same checkpoints on native text, produced by the same evaluation code in the same run;
  comparisons against the other conditions are performed elsewhere.
- The test split is a single held-out set of 490 examples; per-class estimates, especially for
  `severe` (55 cases) and `moderate` (71 cases), have limited statistical power.
- The source datasets state no licence. Trained weights and derived data should not be
  redistributed until that is settled.
