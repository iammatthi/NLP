# Dataset card: derived splits and translation artefacts

Tracked sources (in `data/raw/`, verified by `data/checksums.txt`):

| File | Source | Rows | Licence |
| --- | --- | --- | --- |
| `depression-severity-bengali.xlsx` | Kabir et al. 2022, _JMIR Formative Research_ | 4,897 | not stated |
| `depression-severity-english.csv` | Naseem et al. 2022, _WWW_ | 3,553 | not stated |

Verify with:

```bash
uv run python scripts/verify_data.py
bash scripts/reproduce.sh   # regenerates derived files below
```

## Derived artefacts

### Tracked (checksummed, not regenerated)

- **Translation alignments** (`data/translated/en_to_bn.csv`,
  `bn_to_en.csv`, plus experiment-5 `bn_test_en.csv` / `bn_pool_en.csv`):
  Amazon Translate `TranslateText`, per-row sha256 + label kept beside the
  translation. These are tracked in git and pinned by `data/checksums.txt`.
  (`*.json` metadata and `*.checkpoint.jsonl` are gitignored, regenerate
  on demand and are not checked.)

### Regenerated (gitignored)

- **Splits** (`data/splits/{bn,en}_{train,val,test}.csv`): stratified
  80/10/10, seed 42, produced by `scripts/prepare_splits.py`. Byte-reproducible
  from the sources + seed; the seed is the single source of truth.
  `scripts/prepare_translated_splits.py` validates the translation join
  and writes `data/splits/{en_to_bn,bn_to_en}/manifest.json` with split-file
  hashes, seeds, and counts.

None of the derived files is published as a standalone dataset: they encode
no additional annotation beyond the published sources, and the sources state
no reuse licence. Trained weights are therefore pushed only to private Hub
repos (see `scripts/push_checkpoint.py`).
