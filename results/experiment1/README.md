# Experiment 1: Zero-Shot Cross-Lingual Transfer (EN -> BN)

Fine-tune on **English** data (`en-train`, validated on `en-val`) and evaluate zero-shot on the held-out **native Bengali** test split (`bn-test`), with no Bengali examples seen during training or model selection. Condition 4a (from [Experiment 4](../experiment4/README.md)) is shown alongside as the controlled native-Bengali reference sharing the identical backbone, hyperparameter recipe, and test split.

|            | 1 (zero-shot transfer)                       | 4a (native controlled)                        |
| ---------- | -------------------------------------------- | --------------------------------------------- |
| backbone   | `FacebookAI/xlm-roberta-base`                | `FacebookAI/xlm-roberta-base`                 |
| train / val| `en-train` (n=2842) / `en-val` (n=355)       | `bn-train` (n=3917) / `bn-val` (n=490)        |
| test       | `bn-test` (n=490, native Bengali)            | `bn-test` (n=490, native Bengali)             |
| parameters | ~278M                                        | ~278M                                         |

Setup: 3 seeds (42/43/44), class-weighted cross-entropy (computed from `en-train`), max length 256, lr 2e-5, batch 16, weight decay 0.01. Best checkpoint selected on **English validation macro-F1**, then scored once zero-shot on the held-out **Bengali test** split (n=490). We report two training schedules for Condition 1:
- **1 (4-ep default):** 4 epochs, warmup 0, early-stopping patience 3 — matching the exact training schedule of Condition 4a for a strictly controlled comparison.
- **1 (10-ep tuned):** up to 10 epochs, warmup 100 steps, early-stopping patience 4 — selected on `en-val` to address minority-class under-convergence on the heavily imbalanced English split. The committed `1_xlmr_seed{N}/` artefacts correspond to this tuned 10-epoch run.

All numbers below are mean +/- std across the 3 seeds.

## Artefacts

Each `1_xlmr_seed{N}/` directory holds the two artefacts written by `train.py` and `evaluate.py`:

- `eval_test.json` — full ordinal metric suite on `bn-test`: macro-F1, per-class F1, MAE, QWK, accuracy, the upgrade/ok/downgrade direction matrix, the confusion matrix, the classification report, and bootstrap 95% CIs for macro-F1, QWK and the downgrade rate.
- `train_metrics.json` — English training summary: final train loss, runtime, English train label counts and the inverse-frequency class weights.

Model checkpoints are stored in a private Hugging Face Hub repository (`simonfrnd/nlp_experiment1`) for downstream reuse. The source datasets state no licence, so trained weights are not redistributed publicly.

Driver: [`notebooks/experiment1_zeroshot.ipynb`](../../notebooks/experiment1_zeroshot.ipynb).

## Results

| metric   | 1 Zero-Shot (4-ep default) | 1 Zero-Shot (10-ep tuned) | 4a Native XLM-R   |
| -------- | -------------------------- | ------------------------- | ----------------- |
| macro-F1 | 0.3341 +/- 0.0076          | 0.3571 +/- 0.0142         | 0.7682 +/- 0.0313 |
| QWK      | 0.1900 +/- 0.0434          | 0.2637 +/- 0.0161         | 0.7798 +/- 0.0136 |
| MAE      | 1.0163 +/- 0.0897          | 0.9367 +/- 0.0321         | 0.2707 +/- 0.0290 |
| accuracy | 0.4054 +/- 0.0139          | 0.4272 +/- 0.0031         | 0.8054 +/- 0.0286 |

| error direction                        | 1 Zero-Shot (4-ep default) | 1 Zero-Shot (10-ep tuned) | 4a Native XLM-R   |
| -------------------------------------- | -------------------------- | ------------------------- | ----------------- |
| upgrade rate (pred > gold)             | 0.2741 +/- 0.0745          | 0.2381 +/- 0.0165         | 0.1095 +/- 0.0125 |
| downgrade rate (pred < gold)           | 0.3204 +/- 0.0616          | 0.3347 +/- 0.0184         | 0.0850 +/- 0.0183 |
| error rate                             | 0.5946 +/- 0.0139          | 0.5728 +/- 0.0031         | 0.1946 +/- 0.0286 |
| downgrade share of errors (down / err) | 0.5406 +/- 0.1151          | 0.5843 +/- 0.0301         | 0.4348 +/- 0.0368 |

|                                 | 1 Zero-Shot (4-ep default) | 1 Zero-Shot (10-ep tuned) | 4a Native XLM-R   |
| ------------------------------- | -------------------------- | ------------------------- | ----------------- |
| mean signed error (pred - gold) | 0.0735 +/- 0.2770          | -0.0184 +/- 0.0827        | 0.0381 +/- 0.0139 |

Per-class F1:

| class    | test support | 1 Zero-Shot (4-ep default) | 1 Zero-Shot (10-ep tuned) | 4a Native XLM-R   |
| -------- | ------------ | -------------------------- | ------------------------- | ----------------- |
| minimal  | 210          | 0.5631 +/- 0.0216          | 0.5714 +/- 0.0222         | 0.8790 +/- 0.0169 |
| mild     | 154          | 0.1678 +/- 0.0614          | 0.1242 +/- 0.0856         | 0.8143 +/- 0.0390 |
| moderate | 71           | 0.3314 +/- 0.0400          | 0.3727 +/- 0.0063         | 0.5950 +/- 0.0400 |
| severe   | 55           | 0.2740 +/- 0.0595          | 0.3603 +/- 0.0337         | 0.7845 +/- 0.0311 |

Precision / recall (mean across seeds):

| class    | 1 Zero-Shot (4-ep) P / R | 1 Zero-Shot (10-ep) P / R | 4a Native XLM-R P / R |
| -------- | ------------------------ | ------------------------- | --------------------- |
| minimal  | 0.4997 / 0.6476          | 0.4897 / 0.6857           | 0.8889 / 0.8698       |
| mild     | 0.4044 / 0.1126          | 0.3488 / 0.0779           | 0.8305 / 0.8009       |
| moderate | 0.3577 / 0.3146          | 0.3680 / 0.3803           | 0.5776 / 0.6150       |
| severe   | 0.2138 / 0.4182          | 0.2963 / 0.4788           | 0.7538 / 0.8182       |

Class weights used by the loss (computed on `en-train`, from `train_metrics.json`), compared with `bn-train` in Experiment 4:

| class    | EN train count | EN class weight | BN train count | BN class weight |
| -------- | -------------- | --------------- | -------------- | --------------- |
| minimal  | 2069 (72.8%)   | 0.1560          | 1677 (42.8%)   | 0.4387          |
| mild     | 232 (8.2%)     | 1.3912          | 1232 (31.5%)   | 0.5971          |
| moderate | 315 (11.1%)    | 1.0246          | 566 (14.4%)    | 1.2998          |
| severe   | 226 (8.0%)     | 1.4281          | 442 (11.3%)    | 1.6644          |

Validation trajectory on English (`en-val`, n=355) during training (mean across seeds):
- Under the **4-epoch default schedule** (`patience=3`, `warmup=0`), `load_best_model_at_end=True` selects epoch 4 for seed 42 (0.4744), epoch 3 for seed 43 (0.4286), and epoch 2 for seed 44 (0.5334), with a mean best `en-val` macro-F1 of **0.4788 +/- 0.0525**.
- Under the **10-epoch tuned schedule** (`patience=4`, `warmup=100`), seeds 42 and 43 peak at epoch 4 (0.5130 and 0.5388) and stop early after epoch 8, while seed 44 peaks at epoch 9 (0.5239) and completes 10 epochs, raising mean best `en-val` macro-F1 to **0.5252 +/- 0.0130** (`0.5270 +/- 0.0137` via standalone `evaluate.py`).

| epoch | 1 (4-ep) EN val macro-F1 / loss | 1 (10-ep) EN val macro-F1 / loss | 4a BN val macro-F1 / val loss |
| ----- | ------------------------------- | -------------------------------- | ----------------------------- |
| 1     | 0.2511 / 1.2913                 | 0.2441 / 1.3057                  | 0.5277 / 1.0781               |
| 2     | 0.4254 / 1.1900                 | 0.4132 / 1.2140                  | 0.7392 / 0.6388               |
| 3     | 0.4465 / 1.2473                 | 0.4617 / 1.2213                  | 0.7728 / 0.5760               |
| 4     | 0.4670 / 1.1823                 | 0.5058 / 1.2277                  | 0.7940 / 0.5526               |
| 5     | —                               | 0.4924 / 1.3323                  | —                             |
| 6     | —                               | 0.4979 / 1.5703                  | —                             |
| 7     | —                               | 0.5069 / 1.8287                  | —                             |
| 8     | —                               | 0.4970 / 2.0410                  | —                             |
| 9*    | —                               | 0.5239 / 1.7960                  | —                             |
| 10*   | —                               | 0.5041 / 1.8100                  | —                             |

*\*Epochs 9–10 in the 10-epoch schedule reflect seed 44 only, as seeds 42 and 43 triggered early stopping at epoch 8.*

### 1. Hyperparameter tuning on `en-val` (Seed 42)

Because `en-train` has a 72.8% `minimal` prior (`8.0–11.1%` per minority tier, yielding a ~9x inverse-frequency weight ratio between `severe`/`mild` and `minimal`), the default 4-epoch zero-warmup schedule under-converged on English (`en-val` macro-F1 = 0.4744 on seed 42; 3-seed mean 0.4788 +/- 0.0525). A controlled single-seed sweep on `en-val` (seed 42) tested warmup, training horizon, learning rate, sequence length, and loss weighting.
Adding a 100-step linear warmup and extending training to up to 10 epochs (patience 4) raised `en-val` macro-F1 by +0.0832 on seed 42 while substantially improving ordinal calibration (`QWK = 0.4954`, `MAE = 0.4732`). Removing inverse-frequency class weights caused `mild` F1 to fall from 0.44 to 0.29, confirming that class weighting remains necessary on `en-train`.

### 2. Reconciling `macro-F1` (~0.53–0.56)

Existing literature defines ordinal **Graded Precision (`GP`)** and **Graded Recall (`GR`)** over directional upgrade and downgrade errors ($FP = \text{upgrade rate}$, $FN = \text{downgrade rate}$), which reduce to:

$$\text{GP} = \frac{N_{\text{ok}}}{N_{\text{ok}} + N_{\text{upgrade}}}, \qquad \text{GR} = \frac{N_{\text{ok}}}{N_{\text{ok}} + N_{\text{downgrade}}}, \qquad \text{FScore} = \frac{2 \cdot \text{GP} \cdot \text{GR}}{\text{GP} + \text{GR}}$$


| English split   | macro-F1          | QWK               | MAE               | accuracy          | GP                | GR                | **FScore**        |
| --------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | ----------------- | --------------------- |
| `en-val` (355)  | 0.5270 +/- 0.0137 | 0.4057 +/- 0.0273 | 0.5793 +/- 0.0374 | 0.7099 +/- 0.0149 | 0.7851 +/- 0.0261 | 0.8803 +/- 0.0196 | **0.8296 +/- 0.0099** |
| `en-test` (356) | 0.5553 +/- 0.0257 | 0.4845 +/- 0.0254 | 0.4934 +/- 0.0493 | 0.7434 +/- 0.0218 | 0.8212 +/- 0.0298 | 0.8872 +/- 0.0067 | **0.8527 +/- 0.0144** |

Because 72.8% of the English dataset belongs to `minimal`, overall directional accuracy ($N_{\text{ok}}$) is high (74.34%), yielding a Graded `FScore` of **0.8527 +/- 0.0144** on `en-test`. By contrast, standard `macro-F1` (0.5553 +/- 0.0257 on `en-test`) weights all four severity tiers equally ($1/4$ each), penalising confusion among the small minority tiers.

## Interpretation

### Cross-lingual transfer gap (Condition 1 vs. Condition 4a)

On the held-out native Bengali test split (`bn-test`, n=490), zero-shot transfer from English reaches **0.3341 +/- 0.0076** macro-F1 and **0.1900 +/- 0.0434** QWK under the default 4-epoch recipe, and improves to **0.3571 +/- 0.0142** macro-F1 (+0.0230) and **0.2637 +/- 0.0161** QWK (+0.0737) under the tuned 10-epoch recipe (`MAE` drops from 1.0163 to 0.9367; `accuracy` rises from 0.4054 to 0.4272). While both configurations sit above non-informative baselines on the same split (majority-class macro-F1 0.15, uniform-random ~0.22, QWK 0.00), zero-shot transfer still trails native Bengali fine-tuning of the identical XLM-R backbone (Condition 4a) by **-0.4341 macro-F1** (-0.5898 QWK, +0.7456 MAE) under the matched 4-epoch recipe and by **-0.4111 macro-F1** (-0.5161 QWK, +0.6660 MAE) after source-side hyperparameter tuning — an average error of roughly one full clinical severity tier (0.94–1.02 vs. 0.2707 in 4a).

. Because Condition 1 (4-ep) and Condition 4a share the exact same backbone, hyperparameter recipe, and test examples, the -0.4341 macro-F1 gap (-0.4111 after tuning) isolates the combined cost of replacing native `bn-train` supervision with `en-train`. Several confounded differences between the two training sets contribute to this gap:

1. **Weak source-task separability and sample scarcity in minority tiers.** Even on the English splits, XLM-R reaches a mean best `en-val` macro-F1 of 0.4788 +/- 0.0525 (4-ep) and 0.5252 +/- 0.0130 (10-ep tuned; 0.5553 +/- 0.0257 on `en-test`), compared with 0.7940 on `bn-val` and 0.7682 on `bn-test` in Condition 4a. While inverse-frequency loss weighting equalises the total loss weight ($N_c \cdot w_c$) of each class across the training split, `en-train` contains only 232 `mild`, 315 `moderate`, and 226 `severe` posts against 2069 `minimal` posts (72.8%). Reweighting a small pool of minority examples cannot substitute for lexical diversity.
2. **Language, platform, and annotation shift.** `en-train` consists of longer posts (median ~80 words, DDA-annotated), whereas `bn-test` consists of short Bengali posts (median ~22 words).

### Seed variability and training dynamics

Under the **4-epoch default schedule**, test macro-F1 on `bn-test` is tightly clustered across seeds (0.3341 +/- 0.0076: seed 42 = 0.3408, seed 43 = 0.3259, seed 44 = 0.3355), whereas directional metrics vary substantially more across seeds (`downgrade_rate` +/- 0.0616, `upgrade_rate` +/- 0.0745, `mean_signed_error` +/- 0.2770). Unlike Condition 4a — where validation loss fell monotonically and every seed selected epoch 4 — training on `en-train` without warmup plateaus earlier on `en-val` (selecting epoch 4 for seed 42, epoch 3 for seed 43, and epoch 2 for seed 44). Because `mild` (232 train examples) and `severe` (226) are weighted ~9x more heavily per example than `minimal` (1.3912 and 1.4281 vs. 0.1560), small boundary shifts between epochs move batches of ambiguous Bengali test items between the `minimal` and `severe` endpoints, swinging the signed error (-0.2122 in seed 43 vs. +0.0918 in seed 42 and +0.3408 in seed 44) while leaving aggregate macro-F1 nearly unchanged.

Under the **10-epoch tuned schedule (`warmup=100`, `patience=4`)**, linear warmup prevents early gradient overshooting on the 9x-weighted minority classes, and all three seeds converge to higher and far more consistent checkpoints (`en-val` macro-F1 0.5252 +/- 0.0130 vs. 0.4788 +/- 0.0525). On `bn-test`, macro-F1 rises to 0.3571 +/- 0.0142 (seed 42 = 0.3479, seed 43 = 0.3734, seed 44 = 0.3501), QWK improves from 0.1900 +/- 0.0434 to 0.2637 +/- 0.0161, and seed-to-seed variance on directional metrics drops by ~3–4x (`downgrade_rate` std +/- 0.0184 vs. +/- 0.0616; `upgrade_rate` std +/- 0.0165 vs. +/- 0.0745; `mean_signed_error` std +/- 0.0827 vs. +/- 0.2770).

### Class-level collapse on `mild` and boundary polarization

The class-level breakdown exposes a structural failure mode in zero-shot transfer across both schedules:

| class    | test support | 1 (4-ep) F1       | 1 (4-ep) P / R  | 1 (10-ep) F1      | 1 (10-ep) P / R | 4a Native F1      |
| -------- | ------------ | ----------------- | --------------- | ----------------- | --------------- | ----------------- |
| minimal  | 210          | 0.5631 +/- 0.0216 | 0.4997 / 0.6476 | 0.5714 +/- 0.0222 | 0.4897 / 0.6857 | 0.8790 +/- 0.0169 |
| mild     | 154          | 0.1678 +/- 0.0614 | 0.4044 / 0.1126 | 0.1242 +/- 0.0856 | 0.3488 / 0.0779 | 0.8143 +/- 0.0390 |
| moderate | 71           | 0.3314 +/- 0.0400 | 0.3577 / 0.3146 | 0.3727 +/- 0.0063 | 0.3680 / 0.3803 | 0.5950 +/- 0.0400 |
| severe   | 55           | 0.2740 +/- 0.0595 | 0.2138 / 0.4182 | 0.3603 +/- 0.0337 | 0.2963 / 0.4788 | 0.7845 +/- 0.0311 |

In native Bengali training (4a), `moderate` was the weakest class (0.5950 F1) while `mild` was strong (0.8143 F1). Under zero-shot transfer from English, tuning improves `minimal` (0.5631 to 0.5714), `moderate` (0.3314 to 0.3727), and especially `severe` (0.2740 to 0.3603), whereas **`mild` collapses under both schedules**: 0.1678 F1 (recall 0.1126, 17.3 of 154 posts) in the 4-epoch run and 0.1242 F1 (recall 0.0779, 12.0 of 154 posts) in the 10-epoch run.

Comparing the mean predicted marginal distribution against the true `bn-test` support illustrates this boundary displacement:

- `minimal` (210 true) is predicted 272.3 times (+62.3 cases) in 4-ep and **294.0 times (+84.0 cases)** in 10-ep.
- `mild` (154 true) is predicted only 46.7 times (-107.3 cases) in 4-ep and **31.0 times (-123.0 cases)** in 10-ep.
- `moderate` (71 true) is predicted 62.7 times (-8.3 cases) in 4-ep and **74.0 times (+3.0 cases)** in 10-ep.
- `severe` (55 true) is predicted 108.3 times (+53.3 cases) in 4-ep and **91.0 times (+36.0 cases)** in 10-ep.

Instead of calibrated ordinal thresholds, the zero-shot classifier hollows out the interior (`mild`) and polarizes toward the two endpoints (`minimal` and `severe`). Notably, `mild` (n=232, weight 1.3912) and `severe` (n=226, weight 1.4281) have nearly identical training counts and inverse-frequency weights on `en-train`, yet `severe` is over-predicted (+53.3 cases in 4-ep; +36.0 in 10-ep) while `mild` collapses (-107.3 cases in 4-ep; -123.0 in 10-ep). This asymmetry reflects endpoint versus interior separability: `severe` is an endpoint tier anchored by overt crisis/self-harm markers whose upweighted decision region expands into adjacent tiers, whereas `mild` is an interior tier whose diffuse expressions of unhappiness overlap with `minimal` (2069 diverse English posts). When evaluated zero-shot on short Bengali posts (where `mild` is 31.4% of `bn-test` vs. 8.2% of `en-train`), `mild` is squeezed from below by the broad `minimal` basin and from above by `moderate` and `severe`.

### Error direction and clinical downgrades

Evaluating directional errors requires separating the **overall error rate** from the **directional balance of those errors**:

1. **Absolute downgrade and upgrade rates both rise with total error:** Because `error_rate` nearly triples from 0.1946 in Condition 4a to 0.5946 (4-ep) and 0.5728 (10-ep) in Condition 1, both `downgrade_rate` (0.0850 in 4a vs. 0.3204 [157.0 cases] in 4-ep and **0.3347 [164.0 cases]** in 10-ep) and `upgrade_rate` (0.1095 in 4a vs. 0.2741 [134.3 cases] in 4-ep and **0.2381 [116.7 cases]** in 10-ep) increase substantially. Clinically, roughly one-third of all Bengali test posts have their severity underestimated (`pred < gold`).
2. **Relative error balance shifts toward downgrades — and stabilises across all seeds after tuning:** Conditioning on errors (`downgrade_share_of_errors`), downgrades rise from 0.4348 +/- 0.0368 of all errors in Condition 4a to 0.5406 +/- 0.1151 (4-ep) and **0.5843 +/- 0.0301 (10-ep)** in Condition 1. In the 4-epoch baseline, downgrades dominated in seed 42 (52.9%) and seed 43 (66.1%), whereas upgrades slightly exceeded downgrades in seed 44 (43.1% downgrades, 129 vs. 170) because 59 `minimal` floor cases were predicted as `severe`. In the 10-epoch tuned run, false `minimal -> severe` floor jumps drop from 38.7 to 26.3 on average, and **downgrades outnumber upgrades in every single seed**: 155 vs. 124 (55.6% of errors) in seed 42, 173 vs. 108 (61.6%) in seed 43, and 164 vs. 118 (58.2%) in seed 44.
3. **Interior classes show a consistent downgrade bias across all seeds and schedules:** Restricting attention to the interior classes (`mild` and `moderate`), where errors can occur in either direction without boundary censoring, downgrades dominate in every seed:

| true class | 1 (4-ep) up / down | 1 (10-ep) up / down | 4a Native up / down |
| ---------- | ------------------ | ------------------- | ------------------- |
| mild       | 48.0 / 88.7        | 40.3 / 101.7        | 17.0 / 13.7         |
| moderate   | 12.3 / 36.3        | 10.3 / 33.7         | 9.3 / 18.0          |

In Condition 1, **88.7 of the 154 `mild` cases (57.6%) in 4-ep and 101.7 (66.0%) in 10-ep are downgraded to `minimal`** (89/99/78 in 4-ep; 100/105/100 in 10-ep), accounting for 56.5% (4-ep) and 62.0% (10-ep) of all downgrades across the test set. Among `moderate` cases (71), 36.3 (4-ep) and 33.7 (10-ep) are downgraded on average (28.7 and 30.7 all the way to `minimal`, skipping `mild` entirely).

Finally, while `mean_signed_error` in the 4-epoch run was slightly positive (+0.0735 +/- 0.2770) due to multi-step jumps from the `minimal` floor to `severe` (mean 38.7 cases contributing +3 each), the reduced endpoint overshooting in the 10-epoch tuned run brings `mean_signed_error` to **-0.0184 +/- 0.0827**, aligning the signed error with the raw case counts where **downgrades exceed upgrades under both schedules** (157.0 vs. 134.3 in 4-ep; 164.0 vs. 116.7 in 10-ep).

## Limitations

- **n=3 seeds.** The +/- values are sample standard deviations across three training runs (`seeds 42, 43, 44`), whereas per-run bootstrap 95% CIs over the 490 test items are stored in `eval_test.json` under `ci95`.
- **Source-side validation selection.** Checkpoints are selected on `en-val` macro-F1 to preserve a strictly zero-shot protocol; however, `en-val` performance carries selection bias and does not monotonically track cross-lingual calibration on `bn-test`.
- **Confounded language, domain, and prior shift.** Condition 1 simultaneously varies language (EN vs. BN), platform/text length and training label prior (72.8% vs. 42.8% `minimal`).
