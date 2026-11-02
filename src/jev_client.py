"""TypeSafe AI Jev ('System One') client and clinical rubrics for Condition 6.

Queries both the categorical `Choice` primitive and the ordinal `Score`
primitive in a single non-autoregressive forward pass per social media post.
All domain labels strictly follow `src.schema.LEVELS = (1, 2, 3, 4)`.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.schema import LEVELS, LEVEL_NAMES, NAME_TO_LEVEL, to_domain_label

_REPO_ROOT = Path(__file__).resolve().parents[1]

CHOICE_INSTRUCTIONS = (
    "Classify the depression severity expressed in this social media post into "
    "exactly one of the four clinical severity tiers (minimal, mild, moderate, severe)."
)

SCORE_INSTRUCTIONS = (
    "Rate the ordinal depression severity expressed in this social media post "
    "along the 4-level clinical progression from minimal (non-clinical) to severe (acute crisis)."
)

# 1. Unified DSM-5 rubric grounded in Kabir et al. (2022, JMIR) & unified schema (6a / 6b).
SEVERITY_CHOICE_CRITERIA: dict[str, str] = {
    "minimal": (
        "Tier 1 - Minimal (Non-clinical): Everyday mood fluctuations, situational unhappiness, "
        "ordinary stress, or neutral expression without clinical depressive symptoms."
    ),
    "mild": (
        "Tier 2 - Mild Depression: General depressive feelings, persistent sadness, hopelessness, "
        "loneliness, or low self-esteem, without severe physical/somatic symptoms or self-harm."
    ),
    "moderate": (
        "Tier 3 - Moderate Depression: Clinical depression with physical or somatic symptoms, "
        "substantial functional distress, or need for professional counseling, therapy, or medication."
    ),
    "severe": (
        "Tier 4 - Severe Depression: Acute psychiatric crisis, explicit suicidal ideation, "
        "suicide attempts, self-harm inclinations, or severe comorbid pathology."
    ),
}

SEVERITY_SCORE_CRITERIA: list[str] = [
    SEVERITY_CHOICE_CRITERIA[LEVEL_NAMES[level]] for level in LEVELS
]

# 2. Naseem et al. 2022 DDA / Dreaddit Reddit rubric (6c).
NASEEM_DDA_CHOICE_INSTRUCTIONS = (
    "Classify the depression severity level of this Reddit post according to the "
    "Depressive Disorder Annotation (DDA) scheme (minimal, mild, moderate, severe)."
)

NASEEM_DDA_SCORE_INSTRUCTIONS = (
    "Rate the ordinal depression severity level of this Reddit post along the "
    "4-level Depressive Disorder Annotation (DDA) scale from minimal to severe."
)

NASEEM_DDA_CHOICE_CRITERIA: dict[str, str] = {
    "minimal": (
        "Tier 1 - Minimal: External situational, interpersonal, relationship, or financial/housing "
        "stress (e.g., discussing friends, partners, or life events) without personal depressive symptoms."
    ),
    "mild": (
        "Tier 2 - Mild Depression: Low mood, feeling sad, upset, tearful or crying, mild irritation, "
        "low self-esteem, or difficulty concentrating."
    ),
    "moderate": (
        "Tier 3 - Moderate Depression: High anxiety, panic mode, nightmares, persistent fatigue or "
        "exhaustion, feeling emotionally trapped, or somatic distress interfering with functioning."
    ),
    "severe": (
        "Tier 4 - Severe Depression: Severe depressive breakdown or vegetative symptoms (e.g., not eating, "
        "not sleeping, laying in bed all day, uncontrollable crying, complete despair) or thoughts of death, "
        "self-harm, or killing oneself."
    ),
}

NASEEM_DDA_SCORE_CRITERIA: list[str] = [
    NASEEM_DDA_CHOICE_CRITERIA[LEVEL_NAMES[level]] for level in LEVELS
]

RUBRIC_CONFIGS: dict[str, dict[str, Any]] = {
    "unified": {
        "choice_instructions": CHOICE_INSTRUCTIONS,
        "score_instructions": SCORE_INSTRUCTIONS,
        "choice_criteria": SEVERITY_CHOICE_CRITERIA,
        "score_criteria": SEVERITY_SCORE_CRITERIA,
    },
    "naseem_dda": {
        "choice_instructions": NASEEM_DDA_CHOICE_INSTRUCTIONS,
        "score_instructions": NASEEM_DDA_SCORE_INSTRUCTIONS,
        "choice_criteria": NASEEM_DDA_CHOICE_CRITERIA,
        "score_criteria": NASEEM_DDA_SCORE_CRITERIA,
    },
}


def load_env_key(env_var: str = "TYPESAFE_API_KEY") -> str | None:
    """Return `TYPESAFE_API_KEY` from environment or local `.env` files."""
    val = os.environ.get(env_var)
    if val:
        return val.strip()

    for candidate in (_REPO_ROOT / ".env", _REPO_ROOT.parent / ".env"):
        if candidate.is_file():
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                if k.startswith("export "):
                    k = k[len("export ") :].strip()
                if k == env_var:
                    cleaned = v.strip().strip("'").strip('"')
                    if cleaned:
                        os.environ[env_var] = cleaned
                        return cleaned
    return None


def build_questions(rubric: str = "unified") -> dict[str, Any]:
    """Construct the bundled Choice + Score single-pass question payload for `rubric`."""
    if rubric not in RUBRIC_CONFIGS:
        raise ValueError(f"Unknown rubric {rubric!r}; choices: {sorted(RUBRIC_CONFIGS)}")
    from typesafe_sdk import Choice, Score

    cfg = RUBRIC_CONFIGS[rubric]
    return {
        "severity_choice": Choice(
            instructions=cfg["choice_instructions"],
            criteria=cfg["choice_criteria"],
        ),
        "severity_score": Score(
            instructions=cfg["score_instructions"],
            criteria=cfg["score_criteria"],
        ),
    }


def _extract_choice_probs(raw_probs: Any) -> list[float]:
    """Normalize ChoiceAnswer probabilities into [p_minimal, p_mild, p_moderate, p_severe]."""
    probs = [0.0, 0.0, 0.0, 0.0]
    if isinstance(raw_probs, dict):
        for idx, level in enumerate(LEVELS):
            name = LEVEL_NAMES[level]
            if name in raw_probs:
                probs[idx] = float(raw_probs[name])
    elif isinstance(raw_probs, (list, tuple)) and len(raw_probs) == len(LEVELS):
        probs = [float(x) for x in raw_probs]

    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [0.25, 0.25, 0.25, 0.25]
    return probs


def _extract_score_probs(raw_probs: Any) -> list[float]:
    """Normalize ScoreAnswer probabilities (keyed 0..3 in typesafe_sdk) into [p_1, p_2, p_3, p_4]."""
    probs = [0.0, 0.0, 0.0, 0.0]
    if isinstance(raw_probs, (list, tuple)) and len(raw_probs) == len(LEVELS):
        probs = [float(x) for x in raw_probs]
    elif isinstance(raw_probs, dict):
        # In typesafe_sdk ScoreAnswer.probabilities is dict[int, float] keyed 0..3
        if all(idx in raw_probs for idx in range(len(LEVELS))):
            probs = [float(raw_probs[idx]) for idx in range(len(LEVELS))]
        elif all(str(idx) in raw_probs for idx in range(len(LEVELS))):
            probs = [float(raw_probs[str(idx)]) for idx in range(len(LEVELS))]
        elif all(LEVEL_NAMES[lvl] in raw_probs for lvl in LEVELS):
            probs = [float(raw_probs[LEVEL_NAMES[lvl]]) for lvl in LEVELS]
        elif len(raw_probs) == len(LEVELS):
            probs = [float(v) for v in raw_probs.values()]

    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [0.25, 0.25, 0.25, 0.25]
    return probs


def parse_system_one_response(response: Any, row_idx: int = 0) -> dict[str, Any]:
    """Parse a TypeSafe `system_one` response into standardized domain predictions (1..4)."""
    choices_map = getattr(response, "choices", None) or getattr(response, "answers", {})
    scores_map = getattr(response, "scores", None) or getattr(response, "answers", {})

    choice_ans = choices_map["severity_choice"]
    score_ans = scores_map["severity_score"]

    # 1. Choice primitive extraction
    probs_choice = _extract_choice_probs(getattr(choice_ans, "probabilities", None))
    raw_choice_key = str(getattr(choice_ans, "choice", "")).strip().lower()
    if raw_choice_key in NAME_TO_LEVEL:
        pred_choice = int(NAME_TO_LEVEL[raw_choice_key])
    else:
        pred_choice = int(to_domain_label(int(np.argmax(probs_choice))))
    confidence_choice = float(getattr(choice_ans, "confidence", max(probs_choice)))

    # 2. Score primitive extraction (0..3 in model/SDK space -> 1..4 in domain space)
    raw_score_probs = getattr(score_ans, "probabilities", None)
    raw_score_val = float(getattr(score_ans, "score", 0.0))
    probs_score = _extract_score_probs(raw_score_probs)

    # Expected domain score E[Y] = sum_{k=1..4} k * p_k (equals raw_score_val + 1.0)
    expected_score = float(sum(level * p for level, p in zip(LEVELS, probs_score)))

    # Round expected score to nearest integer ordinal level in {1, 2, 3, 4}
    pred_score_round = int(np.clip(int(np.floor(expected_score + 0.5)), 1, 4))
    pred_score_argmax = int(to_domain_label(int(np.argmax(probs_score))))
    confidence_score = float(getattr(score_ans, "confidence", max(probs_score)))

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "input_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "output_tokens", None) if usage is not None else None
    model_name = str(getattr(response, "model", "jev-latest"))

    return {
        "idx": int(row_idx),
        "model": model_name,
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "pred_choice": pred_choice,
        "probs_choice": probs_choice,
        "confidence_choice": confidence_choice,
        "pred_score_round": pred_score_round,
        "pred_score_argmax": pred_score_argmax,
        "expected_score": expected_score,
        "raw_sdk_score": raw_score_val,
        "probs_score": probs_score,
        "confidence_score": confidence_score,
    }


async def _score_single_row(
    client: Any,
    semaphore: asyncio.Semaphore,
    row_idx: int,
    text: str,
    lang: str,
    questions: dict[str, Any],
    max_retries: int = 5,
) -> dict[str, Any]:
    """Query Jev for one row with concurrency control and exponential backoff."""
    state_payload = {
        "language": "Bengali" if lang == "bn" else "English",
        "post": str(text),
    }
    async with semaphore:
        last_err: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = await client.system_one(
                    state=state_payload,
                    questions=questions,
                )
                return parse_system_one_response(resp, row_idx=row_idx)
            except Exception as exc:
                last_err = exc
                wait_s = min(2.0 ** attempt, 16.0)
                await asyncio.sleep(wait_s)
        raise RuntimeError(
            f"Jev API call failed for row {row_idx} after {max_retries} attempts: {last_err}"
        ) from last_err


async def predict_split_async(
    texts: list[str],
    lang: str,
    cache_path: Path | None = None,
    concurrency: int = 10,
    rubric: str = "unified",
) -> list[dict[str, Any]]:
    """Score all texts in `texts` with resumable JSONL disk caching."""
    api_key = load_env_key("TYPESAFE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TYPESAFE_API_KEY is not set. Set it in the environment or in Repository/.env"
        )

    cached: dict[int, dict[str, Any]] = {}
    if cache_path is not None and cache_path.is_file():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                cached[int(record["idx"])] = record

    missing_indices = [i for i in range(len(texts)) if i not in cached]
    if not missing_indices:
        print(f"  loaded all {len(texts)} predictions from cache: {cache_path}")
        return [cached[i] for i in range(len(texts))]

    if cached:
        print(
            f"  resuming from cache: {len(cached)}/{len(texts)} cached, "
            f"{len(missing_indices)} remaining"
        )

    from typesafe_sdk import AsyncTypeSafeClient

    questions = build_questions(rubric=rubric)
    semaphore = asyncio.Semaphore(concurrency)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    completed = 0
    total_to_run = len(missing_indices)

    async with AsyncTypeSafeClient(api_key=api_key) as client:
        tasks = [
            asyncio.create_task(
                _score_single_row(
                    client=client,
                    semaphore=semaphore,
                    row_idx=idx,
                    text=texts[idx],
                    lang=lang,
                    questions=questions,
                )
            )
            for idx in missing_indices
        ]

        cache_file = (
            cache_path.open("a", encoding="utf-8") if cache_path is not None else None
        )
        try:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                cached[result["idx"]] = result
                if cache_file is not None:
                    cache_file.write(json.dumps(result) + "\n")
                    cache_file.flush()
                completed += 1
                if completed % 50 == 0 or completed == total_to_run:
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    print(
                        f"  scored {completed:4d}/{total_to_run} "
                        f"({elapsed:5.1f}s, {rate:4.1f} req/s)"
                    )
        finally:
            if cache_file is not None:
                cache_file.close()

    return [cached[i] for i in range(len(texts))]


def predict_split(
    texts: list[str],
    lang: str,
    cache_path: Path | None = None,
    concurrency: int = 10,
    rubric: str = "unified",
) -> list[dict[str, Any]]:
    """Synchronous entry point wrapping `predict_split_async`."""
    return asyncio.run(
        predict_split_async(
            texts=texts,
            lang=lang,
            cache_path=cache_path,
            concurrency=concurrency,
            rubric=rubric,
        )
    )
