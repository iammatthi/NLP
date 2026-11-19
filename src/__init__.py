"""Cross-lingual depression severity experiments."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nlp")
except PackageNotFoundError:  # pragma: no cover - editable checkout without install
    __version__ = "0.1.0"

__all__ = ["__version__"]
