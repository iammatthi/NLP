# Experiment 5: LLM Baselines

Zero-/few-shot LLM severity classification on the shared Bengali test split (n = 490).
The headline comparison is `bn` (native Bengali) versus `bn2en` (Bengali translated to
English with `facebook/nllb-200-distilled-600M`); `en` is a sanity check only (Qwen2.5-3B,
zero-shot). Five open-weight instruction models were run locally in 4-bit:
Gemma-3-4B, Qwen2.5-3B, Qwen2.5-1.5B, Llama-3.2-3B, SmolLM2-1.7B.

Zero-shot runs are deterministic: repeating them with other seeds (done for Gemma) gives
identical values, so read them as a single run. Few-shot rows are mean ± s.d. over seeds 42–44.
The table below summarizes completed runs; smoke-test directories are excluded.

## Results

| model | condition | shots/class | n runs | coverage | macro-F1 | QWK | MAE | downgrade rate | upgrade rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `bn` | 0 | 1 | 1.0000 | 0.2320 | 0.1270 | 0.8776 | 0.4612 | 0.1673 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `bn` | 2 | 3 | 1.0000 +/- 0.0000 | 0.0965 +/- 0.0399 | 0.0000 +/- 0.0000 | 1.2184 +/- 0.7282 | 0.1714 +/- 0.1485 | 0.5816 +/- 0.2651 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `bn2en` | 0 | 1 | 1.0000 | 0.2798 | 0.2837 | 0.9245 | 0.3143 | 0.3327 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `bn2en` | 2 | 3 | 1.0000 +/- 0.0000 | 0.2801 +/- 0.0445 | 0.2180 +/- 0.0759 | 0.9612 +/- 0.0569 | 0.2646 +/- 0.2451 | 0.3748 +/- 0.3106 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `bn` | 0 | 1 | 1.0000 | 0.1439 | 0.0842 | 1.1490 | 0.1306 | 0.6898 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `bn` | 2 | 3 | 1.0000 +/- 0.0000 | 0.2417 +/- 0.0795 | 0.1160 +/- 0.1156 | 0.9823 +/- 0.0383 | 0.3122 +/- 0.2283 | 0.3374 +/- 0.2945 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `bn2en` | 0 | 1 | 1.0000 | 0.4016 | 0.3827 | 0.8469 | 0.1000 | 0.5306 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `bn2en` | 2 | 3 | 1.0000 +/- 0.0000 | 0.3763 +/- 0.0415 | 0.4104 +/- 0.0327 | 0.8102 +/- 0.0645 | 0.1401 +/- 0.0741 | 0.4673 +/- 0.1062 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn` | 0 | 1 | 1.0000 | 0.3683 | 0.3997 | 0.8408 | 0.3061 | 0.2531 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn` | 2 | 3 | 1.0000 +/- 0.0000 | 0.2341 +/- 0.0810 | 0.0575 +/- 0.0994 | 0.9762 +/- 0.0317 | 0.3333 +/- 0.2176 | 0.3327 +/- 0.2997 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn2en` | 0 | 1 | 1.0000 | 0.2951 | 0.3849 | 0.9959 | 0.2245 | 0.3898 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn2en` | 2 | 3 | 1.0000 +/- 0.0000 | 0.4622 +/- 0.0274 | 0.4725 +/- 0.0216 | 0.7150 +/- 0.0243 | 0.1354 +/- 0.0153 | 0.4190 +/- 0.0418 |
| `Qwen/Qwen2.5-3B-Instruct` | `en` | 0 | 1 | 1.0000 | 0.2849 | 0.2407 | 1.1011 | 0.0787 | 0.5000 |
| `google/gemma-3-4b-it` | `bn` | 0 | 3 | 1.0000 +/- 0.0000 | 0.3623 +/- 0.0000 | 0.4062 +/- 0.0000 | 0.7469 +/- 0.0000 | 0.1776 +/- 0.0000 | 0.4653 +/- 0.0000 |
| `google/gemma-3-4b-it` | `bn` | 2 | 3 | 1.0000 +/- 0.0000 | 0.3311 +/- 0.0435 | 0.3317 +/- 0.0339 | 1.0510 +/- 0.1019 | 0.0667 +/- 0.0120 | 0.5932 +/- 0.0562 |
| `google/gemma-3-4b-it` | `bn2en` | 0 | 3 | 1.0000 +/- 0.0000 | 0.4761 +/- 0.0000 | 0.4667 +/- 0.0000 | 0.7122 +/- 0.0000 | 0.1449 +/- 0.0000 | 0.4061 +/- 0.0000 |
| `google/gemma-3-4b-it` | `bn2en` | 2 | 3 | 1.0000 +/- 0.0000 | 0.4173 +/- 0.0378 | 0.4148 +/- 0.0406 | 0.8422 +/- 0.0840 | 0.1061 +/- 0.0230 | 0.4878 +/- 0.0601 |
| `meta-llama/Llama-3.2-3B-Instruct` | `bn` | 0 | 1 | 1.0000 | 0.2897 | 0.2689 | 0.9939 | 0.2020 | 0.4776 |
| `meta-llama/Llama-3.2-3B-Instruct` | `bn` | 2 | 3 | 1.0000 +/- 0.0000 | 0.1975 +/- 0.1347 | 0.0635 +/- 0.1584 | 1.4224 +/- 0.5649 | 0.1673 +/- 0.1477 | 0.5925 +/- 0.2558 |
| `meta-llama/Llama-3.2-3B-Instruct` | `bn2en` | 0 | 1 | 1.0000 | 0.3202 | 0.3073 | 1.0490 | 0.1531 | 0.5265 |
| `meta-llama/Llama-3.2-3B-Instruct` | `bn2en` | 2 | 3 | 1.0000 +/- 0.0000 | 0.4105 +/- 0.0302 | 0.4452 +/- 0.0293 | 0.7980 +/- 0.0359 | 0.2612 +/- 0.0673 | 0.3156 +/- 0.0752 |

## Candidate smoke tests

These 100-post results are for model screening and are not directly comparable with full 490-post evaluations.

| model | condition | n | coverage | macro-F1 | QWK | MAE | downgrade rate | upgrade rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `google/gemma-3-1b-it` | `bn2en` | 100 | 1.0000 | 0.3003 | 0.2628 | 1.1300 | 0.1500 | 0.5400 |
| `google/gemma-3-1b-it` | `bn` | 100 | 1.0000 | 0.1327 | 0.1091 | 1.5400 | 0.0300 | 0.8100 |
| `meta-llama/Llama-3.2-3B-Instruct` | `bn2en` | 100 | 1.0000 | 0.3566 | 0.3635 | 0.9800 | 0.1200 | 0.4900 |
| `meta-llama/Llama-3.2-3B-Instruct` | `bn` | 100 | 1.0000 | 0.2914 | 0.3745 | 0.8700 | 0.2200 | 0.3700 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `bn2en` | 100 | 1.0000 | 0.3530 | 0.4430 | 0.8500 | 0.1400 | 0.5300 |
| `Qwen/Qwen2.5-1.5B-Instruct` | `bn` | 100 | 1.0000 | 0.1289 | 0.1070 | 1.1900 | 0.1400 | 0.7000 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn2en` | 20 | 1.0000 | 0.2167 | 0.3673 | 1.0500 | 0.2500 | 0.4000 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn` | 20 | 1.0000 | 0.2424 | 0.2248 | 1.0000 | 0.3000 | 0.3000 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `bn2en` | 100 | 1.0000 | 0.3239 | 0.3041 | 0.8800 | 0.2900 | 0.2800 |
| `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `bn` | 100 | 1.0000 | 0.2465 | 0.1287 | 0.8700 | 0.4200 | 0.1600 |

## Overview

- Best translated-Bengali result: Gemma-3-4B zero-shot, macro-F1 `0.4761`, QWK `0.4667`,
  MAE `0.7122`, downgrade rate `0.1449`. Qwen2.5-3B with two translated examples per class
  is statistically tied (`0.4622 +/- 0.0274`; single-run bootstrap uncertainty is about +/-0.05).
- Best native-Bengali result by macro-F1: Qwen2.5-3B zero-shot (`0.3683`). Gemma-3-4B has the
  lowest native MAE (`0.7469`) and downgrade rate among the zero-shot runs (`0.1776`).
- Translation raises zero-shot macro-F1 for four of five models (exception: Qwen2.5-3B).
- Two native examples per class are unstable: SmolLM2 collapses to constant predictions
  (QWK = 0) and some Qwen/Llama seeds fall to the majority-class level (macro-F1 about 0.15).
  See the truncation caveat below before relying on these rows.
- Gemma-3-1B was only smoke-tested (100 posts, weakest of the screened models) and was not
  carried forward. Qwen2.5-7B could not be run on a 4 GB GPU.
- All completed reports have full coverage and zero failed answers.


## Protocol

- Scoring: a fixed prompt defines the four levels; the predicted level is the most probable
  digit among the next-token probabilities of `1`–`4` (no free-text generation).
- Few-shot: 2 examples per class (8 per prompt) from a fixed 100-post training-only pool
  (25 per class); seeds 42–44 vary only the selected demonstrations.
- `bn2en`: test posts and the pool are translated once with NLLB-200-distilled-600M
  (chunks of at most 100 words, beam 4; `scripts/translate_bn_to_en.py`). Experiments 2–3
  use Amazon Translate instead, so translation effects are compared in direction, not size.
- Bootstrap 95% CIs (1000 resamples) are stored in each `eval_test.json`.

## Limitations

- Model comparison was made on the test split with one fixed prompt, so the ranking is descriptive.
- **Prompt length:** `HFScorer` truncates inputs at `max_input_tokens=2048`. Native
  Bengali few-shot prompts (eight Bengali examples plus the query) may exceed this and lose the
  query and answer prompt. Native few-shot rows are provisional until prompt token counts
  are checked.
- English is not a language-only comparison because its source dataset differs from Bengali.
- Smoke-test results use 100 posts and are not comparable with full 490-post results.
