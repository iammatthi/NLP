# Experiment 4 — Native Gold Standard (Bengali)

Condition 4 of the research design: fine-tune and evaluate on **native** Bengali
data, with no translation anywhere in the pipeline. Two conditions share the
identical splits and training recipe; only the backbone differs.

|            | 4a (controlled)                                                  | 4b (specialized)        |
| ---------- | ---------------------------------------------------------------- | ----------------------- |
| backbone   | `FacebookAI/xlm-roberta-base`                                    | `csebuetnlp/banglabert` |
| role       | common controlled backbone, also used for cross-lingual transfer | Bengali-specific oracle |
| parameters | ~278M                                                            | ~102M                   |

Setup: 3 seeds (42/43/44), class-weighted cross-entropy, max length 256, lr
2e-5, batch 16, 4 epochs, weight decay 0.01, early-stopping patience 3. Best
checkpoint selected on **validation macro-F1**, then scored once on the held-out
**test** split (n=490). All numbers below are mean +/- std across the 3 seeds.

## Artefacts

Each `{run}_seed{N}/` directory holds the two artefacts written by
`train.py` and `evaluate.py`:

- `eval_test.json` — full ordinal metric suite for the test split: macro-F1,
  per-class F1, MAE, QWK, accuracy, the upgrade/ok/downgrade direction matrix,
  the confusion matrix, the classification report, and bootstrap 95% CIs for
  macro-F1, QWK and the downgrade rate.
- `train_metrics.json` — training summary: final train loss, runtime, label
  counts and the inverse-frequency class weights.

Driver: [`notebooks/experiment4_native.ipynb`](../../notebooks/experiment4_native.ipynb).

## Results

| metric   | 4a XLM-R          | 4b BanglaBERT     |
| -------- | ----------------- | ----------------- |
| macro-F1 | 0.7682 +/- 0.0313 | 0.8130 +/- 0.0013 |
| QWK      | 0.7798 +/- 0.0136 | 0.8287 +/- 0.0084 |
| MAE      | 0.2707 +/- 0.0290 | 0.2204 +/- 0.0108 |
| accuracy | 0.8054 +/- 0.0286 | 0.8313 +/- 0.0101 |

| error direction              | 4a XLM-R          | 4b BanglaBERT     |
| ---------------------------- | ----------------- | ----------------- |
| upgrade rate (pred > gold)   | 0.1095 +/- 0.0125 | 0.0973 +/- 0.0139 |
| downgrade rate (pred < gold) | 0.0850 +/- 0.0183 | 0.0714 +/- 0.0041 |
| error rate                   | 0.1946 +/- 0.0286 | 0.1687 +/- 0.0101 |

|                                 | 4a XLM-R          | 4b BanglaBERT     |
| ------------------------------- | ----------------- | ----------------- |
| mean signed error (pred - gold) | 0.0381 +/- 0.0139 | 0.0354 +/- 0.0249 |

Per-class F1:

| class    | test support | 4a XLM-R          | 4b BanglaBERT     |
| -------- | ------------ | ----------------- | ----------------- |
| minimal  | 210          | 0.8790 +/- 0.0169 | 0.8741 +/- 0.0184 |
| mild     | 154          | 0.8143 +/- 0.0390 | 0.8322 +/- 0.0187 |
| moderate | 71           | 0.5950 +/- 0.0400 | 0.6865 +/- 0.0136 |
| severe   | 55           | 0.7845 +/- 0.0311 | 0.8590 +/- 0.0257 |

Precision / recall (mean across seeds):

| class    | 4a XLM-R P / R  | 4b BanglaBERT P / R |
| -------- | --------------- | ------------------- |
| minimal  | 0.8889 / 0.8698 | 0.9053 / 0.8460     |
| mild     | 0.8305 / 0.8009 | 0.8059 / 0.8615     |
| moderate | 0.5776 / 0.6150 | 0.6793 / 0.6948     |
| severe   | 0.7538 / 0.8182 | 0.8519 / 0.8667     |

Class weights used by the loss, from `train_metrics.json`:

| class    | train count | class weight |
| -------- | ----------- | ------------ |
| minimal  | 1677        | 0.4387       |
| mild     | 1232        | 0.5971       |
| moderate | 566         | 1.2998       |
| severe   | 442         | 1.6644       |

Validation trajectory during training (mean across seeds). The logs come from
the notebook's `train.py` output; the epoch-4 checkpoint is the one selected and
scored above.

| epoch | 4a val macro-F1 / val loss | 4b val macro-F1 / val loss |
| ----- | -------------------------- | -------------------------- |
| 1     | 0.5277 / 1.0781            | 0.7637 / 0.6155            |
| 2     | 0.7392 / 0.6388            | 0.8135 / 0.5149            |
| 3     | 0.7728 / 0.5760            | 0.8236 / 0.5117            |
| 4     | 0.7940 / 0.5526            | 0.8378 / 0.5305            |

## Interpretation

### Backbone comparison

4b leads on every aggregate measure: macro-F1 0.8130 against 0.7682
(+0.0448), QWK 0.8287 against 0.7798 (+0.0489), and MAE 0.2204 against
0.2707 (-0.0503). Both conditions sit far above chance levels: a
majority-class baseline reaches macro-F1 0.15 and a uniform-random baseline
about 0.22 on this split.

The advantage is not uniform across classes. 4b is ahead on mild (+0.018 F1),
moderate (+0.092) and severe (+0.075), and marginally behind on minimal
(-0.005). The mild and minimal differences are small relative to their own
seed spread (mild +/- 0.0390 for 4a, +/- 0.0187 for 4b; minimal +/- 0.0169 and
+/- 0.0184), so only the moderate and severe gaps are clearly outside
run-to-run noise.

Several mechanisms could produce this ordering, and the current runs do not
separate them:

- **Pretraining alignment.** 4b was pretrained on Bengali alone, so fine-tuning
  is a small adjustment to an already task-aligned representation. 4b reaches
  0.7637 val macro-F1 after a single epoch, against 0.5277 for 4a, which is
  consistent with this reading.
- **Tokenizer coverage.** 4b uses a 32k Bengali-specific vocabulary; 4a draws
  from a shared ~250k multilingual vocabulary and must express Bengali through
  more, and more fragmented, subword units. On short informal posts this can
  cost signal directly.
- **Pretraining objective.** 4b is an ELECTRA-style replaced-token
  discriminator while 4a is a masked-language model. Discriminative pretraining
  often transfers more readily to classification.
- **Capacity versus dataset size.** 4b is roughly a third the size of 4a and is
  trained on 3917 examples. The smaller model may simply be better regularised
  for a dataset of this size, and 4a's larger capacity may be partly redundant.
- **Shared hyperparameters.** Both conditions use the same learning rate,
  schedule and batch size. That recipe may not be equally appropriate for both
  backbones, so part of the gap could reflect tuning asymmetry rather than
  representation quality.

### Seed variability and training dynamics

Test macro-F1 varies by +/- 0.0313 across seeds for 4a and by +/- 0.0013 for 4b,
a difference of roughly 23x. The same ordering holds for MAE and for both error
directions. This describes **reproducibility**: 4b's final solution is largely
insensitive to initialisation, while 4a's is not.

Seed spread is not by itself a measure of how far training progressed. Low
variance means the seeds converged on the same solution; it does not establish
that this solution is a good one, since a model can be consistently wrong. Nor
does high variance establish under-training, although wandering between basins
is one common cause. The validation curves are the more direct evidence:

- 4b starts high (0.7637 val macro-F1 after epoch 1) and its validation loss
  reaches a minimum in epoch 3 (0.5117) before rising again in epoch 4
  (0.5305) in all three seeds, while macro-F1 still edges upward. That is a
  plateau or mild overfitting signature.
- 4a starts much lower (0.5277) and improves steadily throughout. Its
  validation loss falls monotonically across all four epochs (1.0781 to
  0.5526) and the final epoch still contributes +0.011 to +0.028 macro-F1.

Under this schedule 4a appears to have more room to improve than 4b. Other
readings of the low variance are possible. A smaller model can have a smoother
optimisation landscape and therefore lower seed sensitivity regardless of
convergence. The checkpoint protocol also compresses variance: selection is on
validation macro-F1 and in all six runs the final epoch was the best, so every
run reports the epoch-4 state. Finally, some of the consistency may reflect a
ceiling imposed by the data rather than the models — if the source labels carry
annotator disagreement, achievable F1 is capped and different runs may
converge on the same compromise.

The schedule itself is short for both conditions. Gradient norms in 4a ran
roughly 6-44 against a clipping threshold of 1.0 throughout training, so a
longer schedule or a lower learning rate is a reasonable follow-up for either
backbone.

### Class-level performance

Both models are strongest at the ends of the scale and weakest in the middle.

| class    | test support | 4a XLM-R          | 4b BanglaBERT     |
| -------- | ------------ | ----------------- | ----------------- |
| minimal  | 210          | 0.8790 +/- 0.0169 | 0.8741 +/- 0.0184 |
| mild     | 154          | 0.8143 +/- 0.0390 | 0.8322 +/- 0.0187 |
| moderate | 71           | 0.5950 +/- 0.0400 | 0.6865 +/- 0.0136 |
| severe   | 55           | 0.7845 +/- 0.0311 | 0.8590 +/- 0.0257 |

`moderate` is an ordinal interior class, flanked by `mild` and `severe`, so
like `mild` it can lose cases in both directions whereas the endpoint classes
cannot. It is the more exposed of the two interior classes. The confusion
matrices show the resulting two-way traffic: for 4a the mild/moderate pair
exchanges about 16 and 13 cases, and moderate also leaks about 9 cases upward
to severe. `minimal` and `severe` each have a single neighbour and correspondingly
fewer routes to error.

Sample count does not account for the gap. `moderate` has 71 test cases and
`severe` only 55, yet `severe` scores roughly 0.20 F1 higher; the loss also
upweights `moderate` (1.30) almost as strongly as `severe` (1.66). Plausible
contributing factors that the present runs cannot distinguish:

- **Label reliability at the boundaries.** If the source annotations are least
  consistent between adjacent middle categories, `moderate` will be the noisiest
  target regardless of model quality.
- **Construct properties.** Severity is a continuum discretised into four
  levels; the middle of such a scale is typically less separable than its
  extremes.
- **Feature overlap.** Posts in the middle band may share vocabulary with both
  neighbours, so lexical evidence for the correct label is weaker.
- **Class heterogeneity.** `moderate` may aggregate several distinct
  presentations, making it internally more diverse than the endpoint classes.
- **Measurement noise.** With 71 test cases this class has the widest seed
  spread of any class for 4a (+/- 0.040). The gap to the other classes is much
  larger than this, so noise is not the main story, but it inflates the
  apparent difficulty.

4b improves `moderate` more than any other class (+0.092 F1), which is
consistent with its vocabulary helping most where the lexical cues are
subtlest.

### Error direction

Both conditions produce more upgrade errors (predicted severity above gold)
than downgrade errors, and mean signed error is positive in all six runs. The
effect is small in magnitude — about +0.04 levels on a four-point scale.

Part of this asymmetry is structural rather than learned. `minimal` is the
floor of the scale and the largest class (210 test cases), so all of its errors
are upgrades by construction; `severe` is the ceiling and the smallest class
(55), so all of its errors are downgrades. In the mean confusion matrices,
`minimal` alone accounts for about 51% of all upgrades in 4a and about 68% in
4b, while `severe` supplies only about a quarter of all downgrades. Comparing
the two rates therefore overstates any systematic bias.

Looking only at the interior classes, where errors can go in either direction,
the pattern differs by class and by model:

| true class | 4a up / down | 4b up / down |
| ---------- | ------------ | ------------ |
| mild       | 17.0 / 13.7  | 10.7 / 10.7  |
| moderate   | 9.3 / 18.0   | 4.7 / 17.0   |

`mild` is close to balanced, and `moderate` is in fact downgraded more often
than upgraded in both conditions. The net upgrade bias is driven mainly by the
floor class rather than by a uniform tendency to overestimate severity.

The marginal distributions show where each model's boundary is displaced. 4a
slightly over-predicts `moderate` (+5.0 cases) and `severe` (+4.7) and
under-predicts `minimal` (-4.3) and `mild` (-5.3), a mild upward shift toward
the top of the scale. 4b is close to calibrated on `moderate` (+1.7) and
`severe` (+1.0) but under-predicts `minimal` heavily (-13.7) and over-predicts
`mild` (+11.0) — a local displacement at the bottom boundary rather than a
global shift.

The inverse-frequency class weights are one candidate explanation for the
upward displacement: they raise the cost of missing the rare high-severity
classes (weight 1.66 for `severe`, 1.30 for `moderate`) and so raise their
recall at the expense of their precision. The precision/recall pairs are
consistent with this — recall exceeds precision for `moderate` and `severe` in
both models, while `mild` shows the reverse and absorbs cases from below.
Alternatives include the argmax decision rule under weighted training, which
shifts the effective thresholds between classes, and residual effects of the
label prior, which is not resampled. A logit-adjustment ablation would separate
these from the weighting itself.

4b has less error mass in both directions — fewer upgrades (0.0973 against
0.1095) and fewer downgrades (0.0714 against 0.0850) — so it is more accurate
in both directions rather than differently biased.

## Limitations

- **n=3 seeds.** The +/- values are sample standard deviations over three runs
  and describe seed spread, not confidence intervals over test examples.
- Per-run **bootstrap 95% CIs** for macro-F1, QWK and the downgrade rate are
  stored inside each `eval_test.json` under `ci95`.
- No significance test between conditions yet. McNemar comparisons are planned
  once more than one configuration exists to compare against.
- Both conditions share one hyperparameter recipe. The comparison may partly
  reflect tuning asymmetry rather than backbone quality.
- The test split is a single held-out set of 490 examples; per-class estimates,
  especially for `severe` (55 cases), have limited statistical power.
- `accuracy` is reported for completeness but is misleading on this prior —
  `minimal` alone is 43% of the train split. Macro-F1 and QWK are the primary
  measures.
