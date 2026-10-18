# Experiment 4 — Native Gold Standard (Bengali)

Condition 4 of the research design: fine-tune and evaluate on **native** Bengali
data, with no translation anywhere in the pipeline. Two conditions share the
identical splits and training recipe; only the backbone differs.

| | 4a (controlled) | 4b (specialized) |
|---|---|---|
| backbone | `FacebookAI/xlm-roberta-base` | `csebuetnlp/banglabert` |
| role | common controlled backbone, also used for cross-lingual transfer | Bengali-specific oracle |
| parameters | ~278M | ~102M |

Setup: 3 seeds (42/43/44), class-weighted cross-entropy, max length 256, lr
2e-5, batch 16, 4 epochs, weight decay 0.01, early-stopping patience 3. Best
checkpoint selected on **validation macro-F1**, then scored once on the held-out
**test** split (n=490). All numbers below are mean +/- std across the 3 seeds.

## Files

Each `{run}_seed{N}/` directory holds the two artefacts written by
`train.py` and `evaluate.py`:

- `eval_test.json` — full ordinal metric suite for the test split: macro-F1,
  per-class F1, MAE, QWK, accuracy, the upgrade/ok/downgrade direction matrix,
  the confusion matrix, the classification report, and bootstrap 95% CIs for
  macro-F1, QWK and the downgrade rate.
- `train_metrics.json` — training summary: final train loss, runtime, label
  counts and the inverse-frequency class weights.

Checkpoints (`model.safetensors`, tokenizer, config) are **not** stored here.
They are pushed to the private Hub repo
[`iammatthi/nlp-exp4-native-bn`](https://huggingface.co/iammatthi/nlp-exp4-native-bn)
by `scripts/push_checkpoint.py`. Local `final/` exports of the same checkpoints
for each seed are kept outside the repo.

Driver: [`notebooks/experiment4_native.ipynb`](../../notebooks/experiment4_native.ipynb).

## Test results

| metric | 4a XLM-R | 4b BanglaBERT |
|---|---|---|
| macro-F1 | 0.7682 +/- 0.0313 | 0.8130 +/- 0.0013 |
| QWK | 0.7798 +/- 0.0136 | 0.8287 +/- 0.0084 |
| MAE | 0.2707 +/- 0.0290 | 0.2204 +/- 0.0108 |
| accuracy | 0.8054 +/- 0.0286 | 0.8313 +/- 0.0101 |

| error direction | 4a XLM-R | 4b BanglaBERT |
|---|---|---|
| upgrade rate (pred > gold) | 0.1095 +/- 0.0125 | 0.0973 +/- 0.0139 |
| downgrade rate (pred < gold) | 0.0850 +/- 0.0183 | 0.0714 +/- 0.0041 |
| error rate | 0.1946 +/- 0.0286 | 0.1687 +/- 0.0101 |

|  | 4a XLM-R | 4b BanglaBERT |
|---|---|---|
| mean signed error (pred - gold) | 0.0381 +/- 0.0139 | 0.0354 +/- 0.0249 |

Per-class F1:

| class | test support | 4a XLM-R | 4b BanglaBERT |
|---|---|---|---|
| minimal | 210 | 0.8790 +/- 0.0169 | 0.8741 +/- 0.0184 |
| mild | 154 | 0.8143 +/- 0.0390 | 0.8322 +/- 0.0187 |
| moderate | 71 | 0.5950 +/- 0.0400 | 0.6865 +/- 0.0136 |
| severe | 55 | 0.7845 +/- 0.0311 | 0.8590 +/- 0.0257 |

Precision / recall (mean across seeds):

| class | 4a XLM-R P / R | 4b BanglaBERT P / R |
|---|---|---|
| minimal | 0.8889 / 0.8698 | 0.9053 / 0.8460 |
| mild | 0.8305 / 0.8009 | 0.8059 / 0.8615 |
| moderate | 0.5776 / 0.6150 | 0.6793 / 0.6948 |
| severe | 0.7538 / 0.8182 | 0.8519 / 0.8667 |

Class weights used by the loss, from `train_metrics.json`:

| class | train count | class weight |
|---|---|---|
| minimal | 1677 | 0.4387 |
| mild | 1232 | 0.5971 |
| moderate | 566 | 1.2998 |
| severe | 442 | 1.6644 |

Validation trajectory during training (mean across seeds). The logs come from
the notebook's `train.py` output; the epoch-4 checkpoint is the one selected and
scored above.

| epoch | 4a val macro-F1 / val loss | 4b val macro-F1 / val loss |
|---|---|---|
| 1 | 0.5277 / 1.0781 | 0.7637 / 0.6155 |
| 2 | 0.7392 / 0.6388 | 0.8135 / 0.5149 |
| 3 | 0.7728 / 0.5760 | 0.8236 / 0.5117 |
| 4 | 0.7940 / 0.5526 | 0.8378 / 0.5305 |

## Interpretation

### BanglaBERT wins, and is far more stable

4b is ahead on every aggregate: macro-F1 0.8130 vs 0.7682
(+0.0448), QWK 0.8287 vs 0.7798 (+0.0489), and MAE
0.2204 vs 0.2707 (-0.0503). Both are far above chance — on this
split a majority-class baseline scores macro-F1 0.15 and a uniform-random
baseline about 0.22.

The gap is not just accuracy. Test macro-F1 varies by +/- 0.0013 across seeds for
4b against +/- 0.0313 for 4a, a roughly 23x difference.

### Low seed variance does not mean "needs more training"

The smaller standard deviation is a **stability** signal, not a sign of
under-training. Standard deviation across seeds measures how sensitive the final
solution is to the initialisation: high variance means different seeds settle in
different basins, which is what under-converged optimisation tends to look like.
Low variance means all three seeds reached essentially the same place. That is
evidence of *more* convergence, not less.

The validation curves settle the question directly:

- **4b converges much faster.** After one epoch it is already at
  0.7637 val macro-F1, versus
  0.5277 for 4a. A Bengali-specific vocabulary makes
  fine-tuning a small, well-conditioned adjustment rather than a large
  multilingual re-purposing.
- **4b is at or past its plateau.** Its validation loss bottoms out in epoch 3
  and ticks *up* in epoch 4 for all three seeds
  (0.5117 -> 0.5305), while macro-F1
  still edges up. That is a mild plateau / overfitting signature.
- **4a is the one still climbing.** Its validation loss falls monotonically
  through epoch 4 (0.5760 -> 0.5526) and the last
  epoch still buys +0.011 to +0.028 macro-F1.

If any condition would benefit from more training it is 4a, not 4b.

That said, 4 epochs is short for both, and 4a's gradient norms ran roughly 6-44
against `max_grad_norm` 1.0 throughout — heavily clipped. A longer schedule or a
lower learning rate is a reasonable ablation. That is a scheduling question,
though, and the low seed variance is not the evidence for it.

### Per class: 4b is ahead on three of four

4b is better on mild (+0.018 F1), moderate (+0.092) and severe (+0.075). It is
marginally worse on minimal (-0.005), which sits well inside the +/- 0.017 seed
spread of either model and is not a meaningful difference. The one real
exception is a wash.

### `moderate` is the hard class for both

Both models peak on the endpoints and dip in the middle:

| class | test support | 4a XLM-R | 4b BanglaBERT |
|---|---|---|---|
| minimal | 210 | 0.8790 +/- 0.0169 | 0.8741 +/- 0.0184 |
| mild | 154 | 0.8143 +/- 0.0390 | 0.8322 +/- 0.0187 |
| moderate | 71 | 0.5950 +/- 0.0400 | 0.6865 +/- 0.0136 |
| severe | 55 | 0.7845 +/- 0.0311 | 0.8590 +/- 0.0257 |

`moderate` is an **ordinal interior class**, flanked by mild and severe. It
absorbs confusion from both neighbours and loses cases to both, whereas
`minimal` and `severe` have only one neighbour each. The confusion matrices show
this clearly — for 4a the mild/moderate pair exchanges about 16 and 13 cases,
and moderate also leaks about 9 cases up to severe.

Support is not the explanation: `moderate` has 71 test cases and `severe` only
55, yet `severe` scores roughly 0.20 F1 higher. The difficulty is the boundary
position, not the sample count. The loss already upweights `moderate` (weight
1.30) so it is not being under-served either.

4b improves `moderate` the most of any class (+0.092 F1), consistent with its
vocabulary helping most where the lexical cues are subtlest.

### Both models over-predict severity

Upgrade errors (predicting a higher severity than gold) outnumber downgrades in
both conditions and in all six runs; mean signed error is positive everywhere.
This is the expected signature of the **inverse-frequency class weighting**:

| class | train count | class weight |
|---|---|---|
| minimal | 1677 | 0.4387 |
| mild | 1232 | 0.5971 |
| moderate | 566 | 1.2998 |
| severe | 442 | 1.6644 |

Weighting raises recall on the rare high-severity classes at the cost of their
precision, which means over-predicting them. The confusion matrices confirm the
direction: for `moderate` and `severe`, recall exceeds precision in both models
(4a severe 0.754 / 0.818), while `mild` shows the reverse (0.830 / 0.801) and
spills upward. The bias is a deliberate trade for macro-F1 and QWK under the
label-prior mismatch, not a defect.

4b has a smaller error mass in both directions — fewer upgrades
(0.0973 vs 0.1095) and fewer downgrades
(0.0714 vs 0.0850) — so it is
simply more accurate in both directions rather than differently biased.

## Caveats

- **n=3 seeds.** The +/- values are sample standard deviations over three runs;
  they describe seed spread, not a confidence interval over test examples.
- Per-run **bootstrap 95% CIs** for macro-F1, QWK and the downgrade rate are
  inside each `eval_test.json` under `ci95`.
- No significance test between conditions yet. McNemar comparisons are planned
  once there is more than one configuration to compare against.
- `accuracy` is included for completeness but is misleading on this prior —
  `minimal` alone is 43% of the train split. Macro-F1 and QWK are the primary
  measures.
- Checkpoints live in a **private** Hub repo and should stay private: the source
  datasets state no licence, so trained weights should not be published until
  that is settled.
