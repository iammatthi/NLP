"""Class-weighted cross-entropy for the HF Trainer.

The Trainer pops `labels` out of the model inputs and calls
`compute_loss_func(outputs, labels, num_items_in_batch=...)`, so we compute
the loss ourselves.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def weighted_cross_entropy(class_weights: torch.Tensor):
    """Return a `compute_loss_func` closure using fixed per-class weights.

    `class_weights` must be ordered by label id (index 0 -> label 0, ...).
    """
    weights = class_weights.detach().float()

    def compute_loss(
        outputs: Any,
        labels: torch.Tensor | None = None,
        num_items_in_batch: int | torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs["logits"]
        if labels is None:
            raise ValueError(
                "weighted_cross_entropy received labels=None; the Trainer must "
                "pass labels through compute_loss_func."
            )
        labels = labels.to(logits.device).long()
        w = weights.to(logits.device)
        return F.cross_entropy(logits, labels, weight=w)

    return compute_loss
