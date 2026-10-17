"""Upload one training run's artefacts to a private Hugging Face Hub repo.

This is how checkpoints survive a Colab runtime restart. Only the best
checkpoint (`final/`, selected on validation macro-F1) and the small metrics
files are pushed; intermediate `checkpoint-*` folders are deliberately skipped.

The target repo is **not** defaulted here - each experiment names its own.

Example (full run: model + metrics):

    python scripts/push_checkpoint.py --repo-id <user>/<repo> --run-dir outputs/4a_xlmr_seed42

Example (after evaluate.py, to add just the tiny metrics files):

    python scripts/push_checkpoint.py --repo-id <user>/<repo> --run-dir outputs/4a_xlmr_seed42 --metrics-only

Example (see what would be uploaded, no network):

    python scripts/push_checkpoint.py --repo-id <user>/<repo> --run-dir outputs/4a_xlmr_seed42 --dry-run

The repo is created **private** on first use. Keep it private: the source
datasets state no licence, so model weights trained on them should not be
published until that is settled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without installing the project (e.g. `python scripts/x.py` on Colab).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

METRIC_FILES = ("train_metrics.json", "val_predictions.npz")
NESTED_METRIC_FILES = ("final/eval_test.json",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        required=True,
        help="target repo, e.g. 'your-user/your-repo' (created private on first use)",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="e.g. outputs/4a_xlmr_seed42")
    parser.add_argument("--metrics-only", action="store_true", help="push only the small metrics files")
    parser.add_argument("--public", action="store_true", help="make the repo public (default: private)")
    parser.add_argument("--dry-run", action="store_true", help="list actions without contacting the Hub")
    return parser.parse_args()


def collect(run_dir: Path, metrics_only: bool) -> list[tuple[Path, str]]:
    """Return (local_path, path_in_repo) pairs for what should be uploaded."""
    run_name = run_dir.name
    items: list[tuple[Path, str]] = []

    if not metrics_only:
        final = run_dir / "final"
        if not final.is_dir():
            raise FileNotFoundError(f"{final} not found - run train.py first")
        items.append((final, f"{run_name}/final"))

    for rel in METRIC_FILES:
        p = run_dir / rel
        if p.exists():
            items.append((p, f"{run_name}/{rel}"))

    # These live *inside* final/, so they are already covered when the folder
    # is uploaded. Only push them separately in --metrics-only mode.
    if metrics_only:
        for rel in NESTED_METRIC_FILES:
            p = run_dir / rel
            if p.exists():
                items.append((p, f"{run_name}/{rel}"))

    return items


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:6.1f} {unit}" if unit != "B" else f"{n:6.0f} B"
        n /= 1024
    return f"{n:.1f} GB"


def describe(items: list[tuple[Path, str]], metrics_only: bool) -> None:
    total = 0
    for local, remote in items:
        if local.is_dir():
            size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file())
        else:
            size = local.stat().st_size
        total += size
        kind = "dir " if local.is_dir() else "file"
        print(f"  {kind}  {human(size)}  {local}  ->  {remote}")
    print(f"  total {human(total)}"
          + ("  (metrics only)" if metrics_only else "  (model + metrics)"))


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{run_dir} not found")

    if not args.dry_run and not args.repo_id:
        raise SystemExit("--repo-id is required")

    items = collect(run_dir, args.metrics_only)
    if not items:
        raise SystemExit(f"nothing to upload from {run_dir}")

    print(f"run : {run_dir}")
    print(f"repo: {args.repo_id}")
    describe(items, args.metrics_only)

    if args.dry_run:
        print("\ndry run - nothing was uploaded")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    who = api.whoami()
    print(f"\nauthenticated as: {who['name']}")

    api.create_repo(args.repo_id, private=not args.public, exist_ok=True)

    run_name = run_dir.name
    for local, remote in items:
        if local.is_dir():
            commit = api.upload_folder(
                repo_id=args.repo_id,
                folder_path=str(local),
                path_in_repo=remote,
                commit_message=f"Add {run_name} (best checkpoint)",
                ignore_patterns=["checkpoint-*", "**/checkpoint-*"],
            )
        else:
            commit = api.upload_file(
                repo_id=args.repo_id,
                path_or_fileobj=str(local),
                path_in_repo=remote,
                commit_message=f"Add {run_name}: {local.name}",
            )
        print(f"  pushed {remote}")

    print(f"\ndone. repo: https://huggingface.co/{args.repo_id}")
    print(f"  (private={not args.public} - the datasets have no licence, keep it private)")


if __name__ == "__main__":
    main()
