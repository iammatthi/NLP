"""Check prompt token lengths against the HFScorer truncation limit.

Experiment 5 scores each post as a system + few-shot + query chat prompt.
`HFScorer` truncates at `max_input_tokens` (2048 by default) from the right,
which silently cuts off the query post when Bengali prompts are long. This
script renders the exact prompts `run_llm.py` would send for the Bengali test
split and reports length statistics so that case can be caught before running
few-shot `bn` experiments.

Exits non-zero when any prompt exceeds the limit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from src.llm import build_messages, few_shot_examples
from src.schema import SPLITS_DIR, load_split

TRANSLATED_DIR = _REPO_ROOT / "data/translated"


def _render(tokenizer, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        if not messages or messages[0].get("role") != "system":
            raise
        merged = [dict(message) for message in messages[1:]]
        if not merged:
            raise ValueError("chat template fallback requires a user message")
        merged[0]["content"] = messages[0]["content"] + "\n\n" + merged[0]["content"]
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True
        )


def prompt_lengths(
    tokenizer,
    texts: list[str],
    shots: list[tuple[str, int]],
    max_input_tokens: int,
) -> np.ndarray:
    lengths = []
    for text in texts:
        prompt = _render(tokenizer, build_messages(text, shots))
        lengths.append(len(tokenizer(prompt, add_special_tokens=False)["input_ids"]))
    return np.asarray(lengths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF model id for the tokenizer")
    parser.add_argument(
        "--shots", type=int, default=0, help="few-shot examples per level"
    )
    parser.add_argument("--seed", type=int, default=42, help="few-shot sampling seed")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="only check the first N test posts (default: all)",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=2048,
        help="truncation limit in HFScorer (default: 2048)",
    )
    parser.add_argument("--splits-dir", type=Path, default=SPLITS_DIR)
    parser.add_argument("--translated-dir", type=Path, default=TRANSLATED_DIR)
    args = parser.parse_args()

    test = load_split("bn", "test", out_dir=args.splits_dir)
    texts = test["text"].astype(str).tolist()
    if args.limit is not None:
        texts = texts[: args.limit]

    shots_data: list[tuple[str, int]] = []
    if args.shots:
        pool_path = args.translated_dir / "bn_pool_en.csv"
        if not pool_path.exists():
            print(
                f"error: few-shot pool not found at {pool_path}; "
                "run the bn_pool translation first or use --shots 0",
                file=sys.stderr,
            )
            return 1
        pool = pd.read_csv(pool_path)
        pool = pool[["text_bn", "label"]].rename(columns={"text_bn": "text"})
        shots_data = few_shot_examples(pool, args.shots, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    lengths = prompt_lengths(tokenizer, texts, shots_data, args.max_input_tokens)
    overflow = int((lengths > args.max_input_tokens).sum())
    print(
        f"model={args.model} shots={args.shots} n={len(lengths)} "
        f"limit={args.max_input_tokens}"
    )
    print(
        f"  min={lengths.min():5d} median={np.median(lengths):6.0f} "
        f"p95={np.percentile(lengths, 95):6.0f} max={lengths.max():5d} "
        f"overflow={overflow}"
    )
    if overflow:
        print(
            f"warning: {overflow} prompt(s) exceed {args.max_input_tokens} tokens "
            "and would be truncated by HFScorer (query post cut off)",
            file=sys.stderr,
        )
        return 1
    print("  OK: all prompts fit within the limit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
