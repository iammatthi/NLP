"""Prompting, parsing, and local/API scoring for Experiment 5."""

from __future__ import annotations

import re
import time
from typing import Any

import numpy as np
import pandas as pd

from src.schema import LEVELS

SYSTEM_PROMPT = """You are assisting with a research study on depression severity in social media posts.
Rate the severity expressed by the author on this scale:
1 = minimal: no or very few signs; ordinary sadness, stress, or venting.
2 = mild: some depressive symptoms, but daily functioning is mostly intact.
3 = moderate: persistent symptoms that clearly affect daily life, work, or relationships.
4 = severe: intense, pervasive symptoms with major impairment; suicidal thoughts or self-harm may occur.

Answer with one digit only: 1, 2, 3, or 4."""

MAX_POST_CHARS = 2400
MAX_SHOT_CHARS = 600


def _clip(text: str, limit: int) -> str:
    value = str(text).strip()
    return value if len(value) <= limit else value[:limit] + " ..."


def build_messages(text: str, shots: list[tuple[str, int]] | None = None) -> list[dict[str, str]]:
    """Build a stable system/few-shot/query chat conversation."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_text, level in shots or []:
        messages.append({"role": "user", "content": f"Post:\n{_clip(shot_text, MAX_SHOT_CHARS)}"})
        messages.append({"role": "assistant", "content": str(int(level))})
    messages.append({"role": "user", "content": f"Post:\n{_clip(text, MAX_POST_CHARS)}"})
    return messages


def few_shot_examples(pool: pd.DataFrame, k: int, seed: int) -> list[tuple[str, int]]:
    """Select exactly k examples per level, deterministically and without replacement."""
    if k < 0:
        raise ValueError("k must be non-negative")
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, int]] = []
    for level in LEVELS:
        subset = pool[pool["label"].astype(int) == level]
        if len(subset) < k:
            raise ValueError(f"pool has only {len(subset)} examples of level {level}")
        indices = rng.choice(len(subset), size=k, replace=False)
        picked = subset.iloc[indices]
        rows.extend((str(text), int(label)) for text, label in zip(picked["text"], picked["label"]))
    order = rng.permutation(len(rows))
    return [rows[int(index)] for index in order]


_NAME_PATTERNS = {1: "minimal|minimum", 2: "mild", 3: "moderate", 4: "severe"}


def parse_level(text: str) -> int | None:
    """Parse one unambiguous level from a model response."""
    value = str(text).strip().lower()
    digits = set(re.findall(r"(?<!\d)[1-4](?!\d)", value))
    if digits:
        return int(digits.pop()) if len(digits) == 1 else None
    hits = {level for level, pattern in _NAME_PATTERNS.items() if re.search(rf"\b(?:{pattern})\b", value)}
    return hits.pop() if len(hits) == 1 else None


def _token_candidates(tokenizer: Any, level: int) -> list[int]:
    candidates: list[int] = []
    for text in (str(level), f" {level}", f"\n{level}", f"{level}"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 1 and ids[0] not in candidates:
            candidates.append(int(ids[0]))
    return candidates


class HFScorer:
    """Score using next-token probabilities from a causal language model."""

    def __init__(self, model_id: str, load_4bit: bool = True, max_input_tokens: int = 2048) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kwargs: dict[str, Any] = {"device_map": "auto"}
        if load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
        self.max_input_tokens = max_input_tokens
        self.digit_ids = {
            level: _token_candidates(self.tok, level) for level in LEVELS
        }
        if any(not ids for ids in self.digit_ids.values()):
            raise ValueError(f"cannot map levels to single tokenizer tokens: {self.digit_ids}")

    def _render(self, messages: list[dict[str, str]]) -> str:
        try:
            return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            if not messages or messages[0].get("role") != "system":
                raise
            merged = [dict(message) for message in messages[1:]]
            if not merged:
                raise ValueError("chat template fallback requires a user message")
            merged[0]["content"] = messages[0]["content"] + "\n\n" + merged[0]["content"]
            return self.tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    def score(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        torch = self._torch
        prompt = self._render(messages)
        enc = self.tok(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0, -1].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        masses = []
        for level in LEVELS:
            ids = torch.tensor(self.digit_ids[level], device=logits.device)
            masses.append(torch.logsumexp(log_probs[ids], dim=0))
        level_logp = torch.stack(masses)
        probs = torch.softmax(level_logp, dim=0).cpu().numpy()
        return {
            "pred": int(LEVELS[int(probs.argmax())]),
            "probs": [float(value) for value in probs],
            "digit_mass": float(level_logp.exp().sum().cpu()),
            "raw": "",
        }


class APIScorer:
    """Score through an OpenAI-compatible endpoint."""

    def __init__(self, model: str, base_url: str | None, api_key: str, max_retries: int = 5, sleep: float = 0.0) -> None:
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = max_retries
        self.sleep = sleep

    def score(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        raw = ""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=0.0, max_tokens=8
                )
                raw = (response.choices[0].message.content or "").strip()
                break
            except Exception as exc:
                raw = f"ERROR: {exc}"
                time.sleep(2**attempt)
        if self.sleep:
            time.sleep(self.sleep)
        return {"pred": parse_level(raw), "probs": None, "digit_mass": None, "raw": raw}