# Experiment 5: LLM Baselines

Zero-/few-shot LLM severity classification on the shared Bengali test split.
The headline comparison is `bn` versus `bn2en`; `en` is a sanity check only.

The table below summarizes completed runs; smoke-test directories are excluded.

## Results

| model | condition | shots/class | n runs | coverage | macro-F1 | QWK | MAE | downgrade rate | upgrade rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-3B-Instruct` | `bn` | 0 | 1 | 1.0000 | 0.3683 | 0.3997 | 0.8408 | 0.3061 | 0.2531 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn` | 2 | 3 | 1.0000 +/- 0.0000 | 0.2341 +/- 0.0810 | 0.0575 +/- 0.0994 | 0.9762 +/- 0.0317 | 0.3333 +/- 0.2176 | 0.3327 +/- 0.2997 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn2en` | 0 | 1 | 1.0000 | 0.2951 | 0.3849 | 0.9959 | 0.2245 | 0.3898 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn2en` | 2 | 3 | 1.0000 +/- 0.0000 | 0.4622 +/- 0.0274 | 0.4725 +/- 0.0216 | 0.7150 +/- 0.0243 | 0.1354 +/- 0.0153 | 0.4190 +/- 0.0418 |
| `Qwen/Qwen2.5-3B-Instruct` | `en` | 0 | 1 | 1.0000 | 0.2849 | 0.2407 | 1.1011 | 0.0787 | 0.5000 |

## Protocol

- Labels use the shared ordinal scale 1 minimal, 2 mild, 3 moderate, 4 severe.
- Few-shot examples are selected from training data only, balanced with k examples per class.
- Failed/refused answers are reported through coverage and a separate strict report.
- Raw post text is never written to result artefacts.

## Limitations

- LLM results, model IDs, and API availability must be recorded only after actual runs.
- English is not a language-only comparison because its source dataset differs from Bengali.
