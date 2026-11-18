# Experiment 4a — Native Bengali XLM-R

## Setup

XLM-R (`FacebookAI/xlm-roberta-base`) was fine-tuned and evaluated on native
Bengali data. The training set contains 3,917 examples; validation and test each
contain 490 examples. Three runs used seeds 42, 43, and 44.

Training used class-weighted cross-entropy, maximum sequence length 256, a
learning rate of `2e-5`, batch size 16, weight decay 0.01, up to 10 epochs,
100 warmup steps, and early-stopping patience 4. The checkpoint with the best
validation macro-F1 was selected for test evaluation. All three runs completed
10 epochs. Reported aggregate results are mean ± sample standard deviation
across the three seeds.

| Class | Training count | Loss weight |
| --- | ---: | ---: |
| minimal | 1,677 | 0.4387 |
| mild | 1,232 | 0.5971 |
| moderate | 566 | 1.2998 |
| severe | 442 | 1.6644 |

## Test results

| Metric | Mean ± SD |
| --- | ---: |
| Macro-F1 | **0.7981 ± 0.0107** |
| QWK | **0.8167 ± 0.0124** |
| MAE | **0.2327 ± 0.0127** |
| Accuracy | **0.8272 ± 0.0085** |

Per-seed aggregate metrics:

| Seed | Macro-F1 | QWK | MAE | Accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 0.7861 | 0.8142 | 0.2367 | 0.8245 |
| 43 | 0.8066 | 0.8301 | 0.2184 | 0.8367 |
| 44 | 0.8016 | 0.8057 | 0.2429 | 0.8204 |

Per-class F1 (test support shown):

| Class | Support | F1, mean ± SD |
| --- | ---: | ---: |
| minimal | 210 | 0.8965 ± 0.0052 |
| mild | 154 | 0.8269 ± 0.0144 |
| moderate | 71 | 0.6203 ± 0.0128 |
| severe | 55 | 0.8488 ± 0.0346 |

Mean test precision / recall across seeds:

| Class | Precision | Recall |
| --- | ---: | ---: |
| minimal | 0.9133 | 0.8810 |
| mild | 0.8269 | 0.8268 |
| moderate | 0.5950 | 0.6526 |
| severe | 0.8498 | 0.8485 |

Ordinal error direction:

| Metric | Mean ± SD |
| --- | ---: |
| Correct predictions | 0.8272 ± 0.0085 |
| Upgrade rate (prediction > label) | 0.1000 ± 0.0201 |
| Downgrade rate (prediction < label) | 0.0728 ± 0.0136 |
| Mean signed error (prediction − label) | 0.0299 ± 0.0325 |

Validation checkpoint selection:

| Seed | Best validation macro-F1 | Epoch |
| ---: | ---: | ---: |
| 42 | 0.8266 | 8 |
| 43 | 0.8151 | 9 |
| 44 | 0.8339 | 8 |

## Artefacts

Each `4a_xlmr_seed{N}/` directory contains `train_metrics.json` and
`eval_test.json`. The test evaluation file includes per-class metrics, confusion
matrix, classification report, ordinal error direction, and bootstrap 95% CIs
for macro-F1, QWK, and downgrade rate. Model weights are not included here.

Driver: [`notebooks/experiment4_native.ipynb`](../../notebooks/experiment4_native.ipynb).
