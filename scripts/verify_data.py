"""Verify the tracked data files against data/checksums.txt.

The two raw source datasets and the four derived translation artefacts are
tracked in git so a fresh clone is self-contained. This script checks that
none of them was silently modified:

    uv run python scripts/verify_data.py

Generated artefacts (data/splits/, *.checkpoint.jsonl, outputs/) are
gitignored and therefore not covered; derived splits carry their own
manifest hashes in data/splits/*/manifest.json.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_CHECKSUMS = _REPO_ROOT / "data/checksums.txt"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(checksums_path: Path, root: Path) -> list[str]:
    failures: list[str] = []
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, _, rel = line.partition("  ")
        path = root / rel.strip()
        if not path.is_file():
            failures.append(f"missing: {rel}")
            continue
        actual = sha256(path)
        if actual != expected.strip():
            failures.append(
                f"mismatch: {rel} (expected {expected.strip()}, got {actual})"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checksums", type=Path, default=DEFAULT_CHECKSUMS)
    args = parser.parse_args()

    failures = verify(args.checksums, _REPO_ROOT)
    if failures:
        for failure in failures:
            print(f"FAILED  {failure}", file=sys.stderr)
        return 1
    print(f"OK: all files in {args.checksums.name} match ({_REPO_ROOT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
