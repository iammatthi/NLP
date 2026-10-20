# Experiment 1: Zero-Shot Cross-Lingual Transfer (EN -> BN)

Fine-tune on **English** data (`en-train`, validated on `en-val`) and evaluate zero-shot on the held-out **native Bengali** test split (`bn-test`), with no Bengali examples seen during training or model selection. Condition 4a (from [Experiment 4](../experiment4/README.md)) is shown alongside as the controlled native-Bengali reference sharing the identical backbone, hyperparameter recipe, and test split.

|            | 1 (zero-shot transfer)                       | 4a (native controlled)                        |
| ---------- | -------------------------------------------- | --------------------------------------------- |
| backbone   | `FacebookAI/xlm-roberta-base`                | `FacebookAI/xlm-roberta-base`                 |
| train / val| `en-train` (n=2842) / `en-val` (n=355)       | `bn-train` (n=3917) / `bn-val` (n=490)        |
| test       | `bn-test` (n=490, native Bengali)            | `bn-test` (n=490, native Bengali)             |
| parameters | ~278M                                        | ~278M                                         |

Setup: 3 seeds (42/43/44), class-weighted cross-entropy (computed from `en-train`), max length 256, lr 2e-5, batch 16, 4 epochs, weight decay 0.01, early-stopping patience 3. Best checkpoint selected on **English validation macro-F1**, then scored once zero-shot on the held-out **Bengali test** split (n=490). All numbers below are mean +/- std across the 3 seeds.

## Artefacts

Each `1_xlmr_seed{N}/` directory holds the two artefacts written by `train.py` and `evaluate.py`:

- `eval_test.json` — full ordinal metric suite on `bn-test`: macro-F1, per-class F1, MAE, QWK, accuracy, the upgrade/ok/downgrade direction matrix, the confusion matrix, the classification report, and bootstrap 95% CIs for macro-F1, QWK and the downgrade rate.
- `train_metrics.json` — English training summary: final train loss, runtime, English train label counts and the inverse-frequency class weights.

Model checkpoints are stored in a private Hugging Face Hub repository (`simonfrnd/nlp_experiment1`) for downstream reuse. The source datasets state no licence, so trained weights are not redistributed publicly.

Driver: [`notebooks/experiment1_zeroshot.ipynb`](../../notebooks/experiment1_zeroshot.ipynb).

## Results

| metric   | 1 Zero-Shot XLM-R | 4a Native XLM-R   |
| -------- | ----------------- | ----------------- |
| macro-F1 | 0.3341 +/- 0.0076 | 0.7682 +/- 0.0313 |
| QWK      | 0.1900 +/- 0.0434 | 0.7798 +/- 0.0136 |
| MAE      | 1.0163 +/- 0.0897 | 0.2707 +/- 0.0290 |
| accuracy | 0.4054 +/- 0.0139 | 0.8054 +/- 0.0286 |

| error direction                        | 1 Zero-Shot XLM-R | 4a Native XLM-R   |
| -------------------------------------- | ----------------- | ----------------- |
| upgrade rate (pred > gold)             | 0.2741 +/- 0.0745 | 0.1095 +/- 0.0125 |
| downgrade rate (pred < gold)           | 0.3204 +/- 0.0616 | 0.0850 +/- 0.0183 |
| error rate                             | 0.5946 +/- 0.0139 | 0.1946 +/- 0.0286 |
| downgrade share of errors (down / err) | 0.5406 +/- 0.1151 | 0.4348 +/- 0.0368 |

|                                 | 1 Zero-Shot XLM-R | 4a Native XLM-R   |
| ------------------------------- | ----------------- | ----------------- |
| mean signed error (pred - gold) | 0.0735 +/- 0.2770 | 0.0381 +/- 0.0139 |

Per-class F1:

| class    | test support | 1 Zero-Shot XLM-R | 4a Native XLM-R   |
| -------- | ------------ | ----------------- | ----------------- |
| minimal  | 210          | 0.5631 +/- 0.0216 | 0.8790 +/- 0.0169 |
| mild     | 154          | 0.1678 +/- 0.0614 | 0.8143 +/- 0.0390 |
| moderate | 71           | 0.3314 +/- 0.0400 | 0.5950 +/- 0.0400 |
| severe   | 55           | 0.2740 +/- 0.0595 | 0.7845 +/- 0.0311 |

Precision / recall (mean across seeds):

| class    | 1 Zero-Shot XLM-R P / R | 4a Native XLM-R P / R |
| -------- | ----------------------- | --------------------- |
| minimal  | 0.4997 / 0.6476         | 0.8889 / 0.8698       |
| mild     | 0.4044 / 0.1126         | 0.8305 / 0.8009       |
| moderate | 0.3577 / 0.3146         | 0.5776 / 0.6150       |
| severe   | 0.2138 / 0.4182         | 0.7538 / 0.8182       |

Class weights used by the loss (computed on `en-train`, from `train_metrics.json`), compared with `bn-train` in Experiment 4:

| class    | EN train count | EN class weight | BN train count | BN class weight |
| -------- | -------------- | --------------- | -------------- | --------------- |
| minimal  | 2069 (72.8%)   | 0.1560          | 1677 (42.8%)   | 0.4387          |
| mild     | 232 (8.2%)     | 1.3912          | 1232 (31.5%)   | 0.5971          |
| moderate | 315 (11.1%)    | 1.0246          | 566 (14.4%)    | 1.2998          |
| severe   | 226 (8.0%)     | 1.4281          | 442 (11.3%)    | 1.6644          |

Validation trajectory on English (`en-val`, n=355) during training (mean across seeds). Because `load_best_model_at_end=True` selects on `en-val` macro-F1, the final checkpoint comes from epoch 4 for seed 42 (0.4744), epoch 3 for seed 43 (0.4286), and epoch 2 for seed 44 (0.5334), with a mean best `en-val` macro-F1 of 0.4788 +/- 0.0525.

| epoch | 1 EN val macro-F1 / val loss | 4a BN val macro-F1 / val loss |
| ----- | ---------------------------- | ----------------------------- |
| 1     | 0.2511 / 1.2913              | 0.5277 / 1.0781               |
| 2     | 0.4254 / 1.1900              | 0.7392 / 0.6388               |
| 3     | 0.4465 / 1.2473              | 0.7728 / 0.5760               |
| 4     | 0.4670 / 1.1823              | 0.7940 / 0.5526               |

## Interpretation

### Cross-lingual transfer gap (Condition 1 vs. Condition 4a)

On the held-out native Bengali test split (`bn-test`, n=490), zero-shot transfer from English reaches 0.3341 +/- 0.0076 macro-F1 and 0.1900 +/- 0.0434 QWK. While this sits above non-informative baselines on the same split (majority-class macro-F1 0.15, uniform-random ~0.22, QWK 0.00), it trails native Bengali fine-tuning of the identical XLM-R backbone (Condition 4a) by -0.4341 macro-F1, -0.5898 QWK, and +0.7456 MAE (1.0163 against 0.2707, an average error of roughly one full clinical severity tier).

Because Condition 1 and Condition 4a share the exact same backbone, hyperparameter recipe, and test examples, the -0.4341 macro-F1 gap isolates the combined cost of replacing native `bn-train` supervision with `en-train`. Several confounded differences between the two training sets contribute to this gap:

1. **Weak source-task separability and sample scarcity in minority tiers.** Even on the English validation split (`en-val`, n=355), XLM-R reaches a mean best macro-F1 of only 0.4788 +/- 0.0525, compared with 0.7940 on `bn-val` in Condition 4a (note that `en-val` and `bn-test` are distinct datasets and not directly subtractable, and `en-val` reflects the checkpoint-selection maximum). While inverse-frequency loss weighting equalises the total loss weight ($N_c \cdot w_c$) of each class across the training split, `en-train` contains only 232 `mild`, 315 `moderate`, and 226 `severe` posts against 2069 `minimal` posts (72.8%). Reweighting a small pool of minority examples cannot substitute for lexical diversity.
2. **Language, platform, and annotation shift.** `en-train` consists of longer English posts (median ~80 words, BDI-derived labels), whereas `bn-test` consists of short Bengali posts (median ~22 words, DSM-5 counselor annotations).

### Seed variability and training dynamics

Test macro-F1 on `bn-test` is tightly clustered across seeds (0.3341 +/- 0.0076: seed 42 = 0.3408, seed 43 = 0.3259, seed 44 = 0.3355), whereas directional metrics vary substantially more across seeds (`downgrade_rate` +/- 0.0616, `upgrade_rate` +/- 0.0745, `mean_signed_error` +/- 0.2770).

Unlike Condition 4a — where validation loss fell monotonically and every seed selected epoch 4 — training on `en-train` plateaus earlier on `en-val`. Because evaluation is run once per epoch over a 4-epoch schedule, `early_stopping_patience=3` never halts training early; all three seeds complete 4 epochs, and `load_best_model_at_end=True` retroactively restores the best checkpoint on `en-val` macro-F1 (epoch 4 for seed 42, epoch 3 for seed 43, epoch 2 for seed 44). Because `mild` (232 train examples) and `severe` (226) are weighted ~9x more heavily per example than `minimal` (1.3912 and 1.4281 vs. 0.1560), the model rapidly fits the small minority classes on English; small boundary shifts between epochs move batches of ambiguous Bengali test items between the `minimal` and `severe` endpoints, swinging the signed error (-0.2122 in seed 43 vs. +0.0918 in seed 42 and +0.3408 in seed 44) while leaving aggregate macro-F1 nearly unchanged.

### Class-level collapse on `mild` and boundary polarization

The class-level breakdown exposes a structural failure mode in zero-shot transfer:

| class    | test support | 1 Zero-Shot F1    | 1 Zero-Shot P / R | 4a Native F1      |
| -------- | ------------ | ----------------- | ----------------- | ----------------- |
| minimal  | 210          | 0.5631 +/- 0.0216 | 0.4997 / 0.6476   | 0.8790 +/- 0.0169 |
| mild     | 154          | 0.1678 +/- 0.0614 | 0.4044 / 0.1126   | 0.8143 +/- 0.0390 |
| moderate | 71           | 0.3314 +/- 0.0400 | 0.3577 / 0.3146   | 0.5950 +/- 0.0400 |
| severe   | 55           | 0.2740 +/- 0.0595 | 0.2138 / 0.4182   | 0.7845 +/- 0.0311 |

In native Bengali training (4a), `moderate` was the weakest class (0.5950 F1) while `mild` was strong (0.8143 F1). Under zero-shot transfer from English, **`mild` collapses to 0.1678 F1 with a mean recall of just 0.1126** (identifying only 17.3 of the 154 `mild` Bengali posts on average).

Comparing the mean predicted marginal distribution against the true `bn-test` support illustrates this boundary displacement:

- `minimal` (210 true) is predicted 272.3 times on average (**+62.3 cases**).
- `mild` (154 true) is predicted only 46.7 times on average (**-107.3 cases**).
- `moderate` (71 true) is predicted 62.7 times on average (**-8.3 cases**).
- `severe` (55 true) is predicted 108.3 times on average (**+53.3 cases**).

Instead of calibrated ordinal thresholds, the zero-shot classifier hollows out the interior (`mild`) and polarizes toward the two endpoints (`minimal` and `severe`). Notably, `mild` (n=232, weight 1.3912) and `severe` (n=226, weight 1.4281) have nearly identical training counts and inverse-frequency weights on `en-train`, yet `severe` is over-predicted (+53.3 cases) while `mild` collapses (-107.3 cases). This asymmetry reflects endpoint versus interior separability: `severe` is an endpoint tier anchored by overt crisis/self-harm markers whose upweighted decision region expands into adjacent tiers, whereas `mild` is an interior tier whose diffuse expressions of unhappiness overlap with `minimal` (2069 diverse English posts). When evaluated zero-shot on short Bengali posts (where `mild` is 31.4% of `bn-test` vs. 8.2% of `en-train`), `mild` is squeezed from below by the broad `minimal` basin (recall 0.6476, precision 0.4997) and from above by `moderate` and `severe` (recall 0.4182, precision 0.2138).

### Error direction and clinical downgrades

Evaluating directional errors requires separating the **overall error rate** from the **directional balance of those errors**:

1. **Absolute downgrade and upgrade rates both rise with total error:** Because `error_rate` triples from 0.1946 in Condition 4a to 0.5946 in Condition 1, both `downgrade_rate` (0.0850 to 0.3204, 157.0 of 490 cases) and `upgrade_rate` (0.1095 to 0.2741, 134.3 of 490 cases) increase substantially. Clinically, nearly one-third of all Bengali test posts have their severity underestimated (`pred < gold`).
2. **Relative error balance shifts toward downgrades, but with high seed variance at the floor:** Conditioning on errors (`downgrade_share_of_errors`), downgrades rise from 0.4348 +/- 0.0368 of all errors in Condition 4a to 0.5406 +/- 0.1151 in Condition 1. Across individual seeds, downgrades dominate in seed 42 (52.9% of errors) and seed 43 (66.1%), whereas upgrades slightly exceed downgrades in seed 44 (43.1% downgrades, 129 vs. 170) because 59 `minimal` floor cases (which can only err upward by construction) were predicted as `severe`.
3. **Interior classes show a consistent downgrade bias across all seeds:** Restricting attention to the interior classes (`mild` and `moderate`), where errors can occur in either direction without boundary censoring, downgrades dominate in every seed:

| true class | 1 Zero-Shot up / down | 4a Native up / down |
| ---------- | --------------------- | ------------------- |
| mild       | 48.0 / 88.7           | 17.0 / 13.7         |
| moderate   | 12.3 / 36.3           | 9.3 / 18.0          |

In Condition 1, **88.7 of the 154 `mild` cases (57.6%) are downgraded to `minimal`** (89 in seed 42, 99 in seed 43, 78 in seed 44), accounting for 56.5% of all downgrades across the test set. Among `moderate` cases (71), 36.3 are downgraded on average (28.7 all the way to `minimal`, skipping `mild` entirely).

Finally, although `mean_signed_error` is slightly positive on average (+0.0735 +/- 0.2770), this is an artefact of multi-step jumps from the `minimal` floor to `severe` (mean 38.7 `minimal` cases predicted as `severe`, contributing +3 each to the signed sum). In raw case counts, **downgrades (157.0 cases, 32.04%) exceed upgrades (134.3 cases, 27.41%)**.

## Limitations

- **n=3 seeds.** The +/- values are sample standard deviations across three training runs (`seeds 42, 43, 44`), whereas per-run bootstrap 95% CIs over the 490 test items are stored in `eval_test.json` under `ci95`.
- **Source-side validation selection.** Checkpoints are selected on `en-val` macro-F1 to preserve a strictly zero-shot protocol; however, `en-val` performance carries selection bias and does not monotonically track cross-lingual calibration on `bn-test`.
- **Confounded language, domain, and prior shift.** Condition 1 simultaneously varies language (EN vs. BN), platform/text length and training label prior (72.8% vs. 42.8% `minimal`).