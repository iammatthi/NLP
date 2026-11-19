"""Tests for the tracked-data integrity checker (scripts/verify_data.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.verify_data import sha256, verify


def test_sha256_and_verify_round_trip(tmp_path):
    data = tmp_path / "data" / "raw" / "x.csv"
    data.parent.mkdir(parents=True)
    data.write_text("hello\n", encoding="utf-8")
    checksums = tmp_path / "checksums.txt"
    checksums.write_text(f"{sha256(data)}  data/raw/x.csv\n", encoding="utf-8")
    assert verify(checksums, tmp_path) == []


def test_verify_reports_missing_and_mismatch(tmp_path):
    data = tmp_path / "data" / "raw" / "x.csv"
    data.parent.mkdir(parents=True)
    data.write_text("hello\n", encoding="utf-8")
    good = sha256(data)
    checksums = tmp_path / "checksums.txt"
    checksums.write_text(
        f"{good}  data/raw/x.csv\n{'0' * 64}  data/raw/absent.csv\n",
        encoding="utf-8",
    )
    data.write_text("tampered\n", encoding="utf-8")
    failures = verify(checksums, tmp_path)
    assert len(failures) == 2
    assert any("mismatch" in f for f in failures)
    assert any("missing" in f for f in failures)
