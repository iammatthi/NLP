"""CPU-only build check for the experiment-4 backbones.

Loads each tokenizer + `AutoModelForSequenceClassification`, runs a real
forward pass on two Bengali rows from the dataset, and asserts the logits
shape. No training, no GPU. Exits non-zero on failure so it can gate CI
and be run before spending Colab time.

Example (both experiment-4 backbones, the default):

    uv run python scripts/smoke_test_model.py

Example (a single model, more probe rows):

    uv run python scripts/smoke_test_model.py --models csebuetnlp/banglabert --rows 4

Example (a different backbone):

    uv run python scripts/smoke_test_model.py --models sagorsarker/bangla-bert-base
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.schema import load_bengali

EXPERIMENT_4_BACKBONES = {
    "4a": "FacebookAI/xlm-roberta-base",
    "4b": "csebuetnlp/banglabert",
}


def check_model(model_id: str, texts: list[str], num_labels: int) -> bool:
    print(f"\n=== {model_id} ===")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=num_labels
        )
        model.eval()
    except Exception:
        traceback.print_exc()
        return False

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  class       : {model.__class__.__name__}")
    print(f"  tokenizer   : {tokenizer.__class__.__name__} (vocab={tokenizer.vocab_size})")
    print(f"  parameters  : {n_params:,} ({n_trainable:,} trainable)")

    try:
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            out = model(**enc)
        logits = out.logits
    except Exception:
        traceback.print_exc()
        return False

    print(f"  tokens      : {enc['input_ids'].shape}")
    print(f"  logits      : {tuple(logits.shape)}")

    expected = (len(texts), num_labels)
    if tuple(logits.shape) != expected:
        print(f"  FAIL: expected logits {expected}, got {tuple(logits.shape)}")
        return False
    if not torch.isfinite(logits).all():
        print("  FAIL: non-finite logits")
        return False

    print("  OK")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-labels", type=int, default=4)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="HF model ids (default: the experiment-4 backbones)",
    )
    args = parser.parse_args()

    frame = load_bengali()
    texts = frame["text"].head(args.rows).tolist()
    print(f"probe texts: {len(texts)} Bengali rows from the dataset")

    model_ids = args.models or list(EXPERIMENT_4_BACKBONES.values())
    results = {mid: check_model(mid, texts, args.num_labels) for mid in model_ids}

    print("\n=== summary ===")
    for mid, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {mid}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
