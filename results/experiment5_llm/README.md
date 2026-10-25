# Experiment 5: LLM Baselines

Zero-/few-shot LLM severity classification on the shared Bengali test split.
The headline comparison is `bn` versus `bn2en`; `en` is a sanity check only.

Results remain pending until real Colab runs are completed.

## Results

| model | condition | shots/class | n runs | coverage | macro-F1 | QWK | MAE | downgrade rate | upgrade rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `Qwen/Qwen2.5-3B-Instruct` | `bn` | 0 | 2 | 1.0000 +/- 0.0000 | 0.3054 +/- 0.0890 | 0.3122 +/- 0.1237 | 0.9204 +/- 0.1126 | 0.3031 +/- 0.0043 | 0.2765 +/- 0.0332 |
| `Qwen/Qwen2.5-3B-Instruct` | `bn2en` | 0 | 2 | 1.0000 +/- 0.0000 | 0.2559 +/- 0.0555 | 0.3761 +/- 0.0125 | 1.0230 +/- 0.0382 | 0.2372 +/- 0.0180 | 0.3949 +/- 0.0072 |

## Protocol

- Labels use the shared ordinal scale 1 minimal, 2 mild, 3 moderate, 4 severe.
- Few-shot examples are selected from training data only, balanced with k examples per class.
- Failed/refused answers are reported through coverage and a separate strict report.
- Raw post text is never written to result artefacts.

## Limitations

- LLM results, model IDs, and API availability must be recorded only after actual runs.
- English is not a language-only comparison because its source dataset differs from Bengali.
