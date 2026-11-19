#!/usr/bin/env bash
# Regenerate the derived data artefacts and verify reproducibility.
# Training/evaluation themselves need a GPU and are driven by the notebooks
# or `scripts/reproduce.sh --full` (see below); this script covers everything
# that must be byte-reproducible locally.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "${1:-}" == "--full" ]]; then
  echo "== install (uv, dev tools) =="
  uv sync --dev
fi

echo "== verify tracked data files =="
uv run python scripts/verify_data.py

echo "== prepare splits (bn + en, seed 42) =="
uv run python scripts/prepare_splits.py --lang bn --seed 42
uv run python scripts/prepare_splits.py --lang en --seed 42

echo "== prepare translated splits (en_to_bn + bn_to_en, seed 42) =="
uv run python scripts/prepare_translated_splits.py --lang en --seed 42
uv run python scripts/prepare_translated_splits.py --lang bn --seed 42

echo "== CPU build check (tokenizer + forward pass, no training) =="
uv run python scripts/smoke_test_model.py --models FacebookAI/xlm-roberta-base

echo
echo "Data pipeline is reproducible from the tracked sources."
echo "Next: train/evaluate, e.g."
echo "  uv run python train.py --model xlmr --lang bn --output-dir outputs/4a_xlmr_seed42 --seed 42"
echo "  uv run python evaluate.py --model-dir outputs/4a_xlmr_seed42/final --lang bn --split test"
echo "Or run the corresponding notebook end-to-end (Colab). See README.md."
