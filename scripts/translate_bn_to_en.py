"""Translate the Bengali test split and fixed few-shot pool with NLLB."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd

from src.schema import LEVELS, load_split

OUT_DIR = Path("data/translated")
MAX_WORDS_PER_CHUNK = 100


def split_post(text: str) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"(?<=[।.!?])\s+|\n+", str(text)) if piece.strip()]
    chunks: list[str] = []
    for piece in pieces:
        words = piece.split()
        chunks.extend(" ".join(words[start : start + MAX_WORDS_PER_CHUNK]) for start in range(0, len(words), MAX_WORDS_PER_CHUNK))
    return chunks or [str(text)]


def translate_posts(texts: list[str], model_id: str, batch_size: int, beams: int) -> list[str]:
    import torch
    from tqdm import tqdm
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id, src_lang="ben_Beng")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()
    if device == "cuda":
        model = model.half()
    english_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    flat = [(post_index, chunk_index, chunk) for post_index, text in enumerate(texts) for chunk_index, chunk in enumerate(split_post(text))]
    order = sorted(range(len(flat)), key=lambda index: len(flat[index][2]))
    translations = [""] * len(flat)
    for start in tqdm(range(0, len(order), batch_size), desc="translating"):
        indices = order[start : start + batch_size]
        encoded = tokenizer([flat[index][2] for index in indices], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        with torch.no_grad():
            generated = model.generate(**encoded, forced_bos_token_id=english_id, num_beams=beams, max_new_tokens=256)
        for index, text in zip(indices, tokenizer.batch_decode(generated, skip_special_tokens=True)):
            translations[index] = text
    grouped: list[list[tuple[int, str]]] = [[] for _ in texts]
    for (post_index, chunk_index, _), translation in zip(flat, translations):
        grouped[post_index].append((chunk_index, translation))
    return [" ".join(text for _, text in sorted(parts)) for parts in grouped]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--pool-per-class", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        raise ValueError("limit must be non-negative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    test = load_split("bn", "test")
    train = load_split("bn", "train")
    pool = pd.concat([train[train["label"] == level].sample(n=args.pool_per_class, random_state=0) for level in LEVELS])
    pool = pool.rename(columns={"text": "text_bn"}).rename_axis("row_id").reset_index()
    test_out = test.rename(columns={"text": "text_bn"}).copy()
    if args.limit is not None:
        test_out = test_out.head(args.limit)
    test_out["text_en"] = translate_posts(test_out["text_bn"].astype(str).tolist(), args.model, args.batch_size, args.beams)
    pool["text_en"] = translate_posts(pool["text_bn"].astype(str).tolist(), args.model, args.batch_size, args.beams)
    suffix = "_smoke" if args.limit is not None else ""
    test_out.to_csv(args.out_dir / f"bn_test_en{suffix}.csv", index=False)
    pool.to_csv(args.out_dir / f"bn_pool_en{suffix}.csv", index=False)


if __name__ == "__main__":
    main()