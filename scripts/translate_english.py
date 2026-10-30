"""Translate the full English dataset to Bengali with Amazon Translate.

Run with --dry-run to inspect the workload without making API calls. Completed
rows are checkpointed locally so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.schema import ENGLISH_PATH, load_english

DEFAULT_OUTPUT = Path("data/translated")
MAX_REQUEST_BYTES = 10_000


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_checkpoint(path: Path, frame) -> dict[int, dict]:
    if not path.exists():
        return {}
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
                idx = record["source_row"]
                if type(idx) is not int or not 0 <= idx < len(frame) or idx in records:
                    raise ValueError("invalid or duplicate source row")
                if record["source_sha256"] != text_hash(frame.iloc[idx]["text"]):
                    raise ValueError("source text has changed")
                if record["label"] != int(frame.iloc[idx]["label"]):
                    raise ValueError("source label has changed")
                if not isinstance(record["region"], str) or not record["region"]:
                    raise ValueError("missing region")
                if not isinstance(record["text"], str) or not record["text"].strip():
                    raise ValueError("empty translation")
                records[idx] = record
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid checkpoint at line {line_number}: {exc}") from exc
    return records


def translate(frame, client, checkpoint: Path, region: str) -> dict[int, dict]:
    records = load_checkpoint(checkpoint, frame)
    if any(record["region"] != region for record in records.values()):
        raise ValueError("Checkpoint region differs from the requested region")
    with checkpoint.open("a", encoding="utf-8") as handle:
        for idx, row in frame.iterrows():
            if idx in records:
                continue
            response = client.translate_text(
                Text=row["text"], SourceLanguageCode="en", TargetLanguageCode="bn"
            )
            translated = response["TranslatedText"]
            if not isinstance(translated, str) or not translated.strip():
                raise ValueError(f"Empty translation returned for source row {idx}")
            record = {
                "source_row": int(idx),
                "source_sha256": text_hash(row["text"]),
                "label": int(row["label"]),
                "text": translated,
                "region": region,
                "translated_at": datetime.now(timezone.utc).isoformat(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            records[idx] = record
            if len(records) % 100 == 0 or len(records) == len(frame):
                print(f"Translated {len(records)}/{len(frame)} rows", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ENGLISH_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    frame = load_english(args.source)
    characters = sum(len(text) for text in frame["text"])
    too_long = [
        idx for idx, text in enumerate(frame["text"])
        if len(text.encode("utf-8")) > MAX_REQUEST_BYTES
    ]
    if too_long:
        raise ValueError(f"Rows exceed Amazon Translate's 10,000-byte limit: {too_long}")
    print(f"Rows: {len(frame)}; input characters: {characters:,}; region: {args.region}")
    print("Standard text translation; free-tier availability depends on your AWS account.")
    if args.dry_run:
        return

    import boto3
    from botocore.config import Config

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "en_to_bn.checkpoint.jsonl"
    output = args.output_dir / "en_to_bn.csv"
    metadata = args.output_dir / "en_to_bn.metadata.json"
    if output.exists() and not checkpoint.exists():
        raise FileExistsError(
            f"{output} exists but its checkpoint is missing; refusing to overwrite"
        )
    client = boto3.client(
        "translate", region_name=args.region,
        config=Config(retries={"mode": "standard", "max_attempts": 5}),
    )
    records = translate(frame, client, checkpoint, args.region)
    result = frame.rename(columns={"text": "source_text"}).copy()
    result.insert(0, "source_row", range(len(result)))
    result["source_sha256"] = [records[idx]["source_sha256"] for idx in range(len(result))]
    result["text"] = [records[idx]["text"] for idx in range(len(result))]
    temporary = output.with_suffix(".csv.tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(output)
    manifest = {
        "provider": "Amazon Translate",
        "operation": "TranslateText",
        "source_language": "en",
        "target_language": "bn",
        "region": args.region,
        "source_file_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "source_rows": len(frame),
        "source_characters": characters,
        "model": "AWS managed standard translation (version not exposed by API)",
        "boto3_version": boto3.__version__,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved translations to {output} and metadata to {metadata}")


if __name__ == "__main__":
    main()
