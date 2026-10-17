"""Fine-tune a backbone on the native-language severity splits (experiment 4).

Condition 4a (controlled)     : --model FacebookAI/xlm-roberta-base --lang bn
Condition 4b (specialized)    : --model csebuetnlp/banglabert     --lang bn

Both conditions train and evaluate on the *same* native splits; only the
backbone differs. Class-weighted cross-entropy counteracts the label prior
without resampling. Checkpoints are selected on validation macro-F1.

Example (experiment 4a, run after `uv run python scripts/prepare_splits.py --lang bn`):

    uv run python train.py --model xlmr --lang bn --output-dir outputs/4a_xlmr \
        --seed 42 --epochs 4 --train-batch-size 16 --learning-rate 2e-5

Example (experiment 4b):

    uv run python train.py --model banglabert --lang bn --output-dir outputs/4b_banglabert \
        --seed 42 --epochs 4 --train-batch-size 16 --learning-rate 2e-5

Smoke-run the training loop on CPU (a few steps, no useful model):

    uv run python train.py --model banglabert --lang bn --output-dir outputs/smoke \
        --max-steps 4 --epochs 1 --train-batch-size 4 --eval-batch-size 8

Repeat with --seed 43 and --seed 44 for the 3-seed runs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Allow running without installing the project, including from a different cwd.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from src.dataset import SeverityDataset
from src.losses import weighted_cross_entropy
from src.metrics import compute_metrics_for_trainer
from src.schema import LEVELS, LEVEL_NAMES, class_weights, load_split, to_domain_label

MODEL_CHOICES = {
    "xlmr": "FacebookAI/xlm-roberta-base",
    "banglabert": "csebuetnlp/banglabert",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF id or alias: xlmr | banglabert")
    parser.add_argument("--lang", choices=("bn", "en"), default="bn")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=float, default=4.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1, help="smoke-run cap")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def resolve_model(name: str) -> str:
    return MODEL_CHOICES.get(name, name)


def main() -> None:
    args = parse_args()
    model_id = resolve_model(args.model)
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.lang, "train")
    val_df = load_split(args.lang, "val")
    print(f"train={len(train_df)}  val={len(val_df)}  model={model_id}")

    labels = train_df["label"].to_numpy()
    counts = {LEVEL_NAMES[lvl]: int((labels == lvl).sum()) for lvl in LEVELS}
    weights = class_weights(labels)
    weight_map = {LEVEL_NAMES[lvl]: round(float(w), 4) for lvl, w in zip(LEVELS, weights)}
    print(f"label counts : {counts}")
    print(f"class weights: {weight_map}" if not args.no_class_weights else "class weights: off")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=len(LEVELS),
        ignore_mismatched_sizes=True,
    )

    train_ds = SeverityDataset(train_df, tokenizer, max_length=args.max_length)
    eval_ds = SeverityDataset(val_df, tokenizer, max_length=args.max_length)

    compute_loss_func = None
    if not args.no_class_weights:
        compute_loss_func = weighted_cross_entropy(torch.tensor(weights, dtype=torch.float32))

    use_cpu = not torch.cuda.is_available()
    if use_cpu and (args.fp16 or args.bf16):
        print("warning: no CUDA available, disabling fp16/bf16")
        args.fp16 = args.bf16 = False

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
        use_cpu=use_cpu,
        fp16=args.fp16,
        bf16=args.bf16,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        compute_loss_func=compute_loss_func,
        compute_metrics=compute_metrics_for_trainer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    result = trainer.train()

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = {k: float(v) for k, v in result.metrics.items()}
    metrics["label_counts"] = counts
    metrics["class_weights"] = weight_map if not args.no_class_weights else None
    metrics["model_id"] = model_id
    metrics["lang"] = args.lang
    metrics["seed"] = args.seed
    (args.output_dir / "train_metrics.json").write_text(json.dumps(metrics, indent=2))

    # Persist raw val logits + gold labels for evaluate.py.
    # Labels are stored in domain space (1..4); argmax(logits) is model space,
    # so `preds` is written out already shifted to domain space.
    pred = trainer.predict(eval_ds)
    logits = pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    np.savez(
        args.output_dir / "val_predictions.npz",
        logits=logits,
        labels=to_domain_label(np.asarray(pred.label_ids)),
        preds=to_domain_label(np.argmax(logits, axis=-1)),
    )

    print(json.dumps(metrics, indent=2))
    print(f"saved model and metrics to {args.output_dir}")


if __name__ == "__main__":
    # Avoid tokenizer-parallelism warnings in the dataloader workers.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
