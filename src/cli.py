"""Installed console entry points for the repo-root experiment scripts.

The training / evaluation entry points live at the repository root
(`train.py`, `evaluate.py`, `run_llm.py`, `evaluate_jev.py`) so the notebooks
and README keep their documented `python train.py ...` invocation. These
thin wrappers make the same entry points available as installed commands:

    uv run nlp-train --help
    uv run nlp-evaluate --help
    uv run nlp-run-llm --help
    uv run nlp-evaluate-jev --help

They require an editable/checkout install (`uv sync`) because they import
the root modules; the repository root directory is added to `sys.path` here.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _import_entry_module(name: str):
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import importlib

    return importlib.import_module(name)


def train() -> None:
    module = _import_entry_module("train")
    module.os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    module.main()


def evaluate() -> None:
    _import_entry_module("evaluate").main()


def run_llm() -> None:
    _import_entry_module("run_llm").main()


def evaluate_jev() -> None:
    _import_entry_module("evaluate_jev").main()
