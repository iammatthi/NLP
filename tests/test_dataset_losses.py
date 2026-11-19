"""Unit tests for the model boundary: dataset labels and weighted loss."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from src.dataset import SeverityDataset
from src.losses import weighted_cross_entropy


class FakeTokenizer:
    def __call__(
        self,
        text,
        truncation=False,
        max_length=256,
        padding=False,
        return_tensors=None,
    ):
        assert isinstance(text, str)
        return {"input_ids": [1, 2, 3]}


def test_dataset_shifts_labels_to_model_space():
    frame = pd.DataFrame({"text": ["a", "b", "c", "d"], "label": [1, 2, 3, 4]})
    dataset = SeverityDataset(frame, FakeTokenizer(), max_length=8)
    assert len(dataset) == 4
    assert dataset.labels.tolist() == [0, 1, 2, 3]
    item = dataset[0]
    assert item["labels"].item() == 0
    assert item["input_ids"].tolist() == [1, 2, 3]


def test_weighted_loss_matches_weighted_cross_entropy():
    weights = torch.tensor([0.5, 1.0, 2.0, 1.5], dtype=torch.float32)
    compute_loss = weighted_cross_entropy(weights)
    logits = torch.randn(6, 4, dtype=torch.float32)
    labels = torch.tensor([0, 1, 2, 3, 0, 2], dtype=torch.long)
    expected = F.cross_entropy(logits, labels, weight=weights.float())
    assert float(compute_loss(SimpleNamespace(logits=logits), labels)) == pytest.approx(
        float(expected)
    )
    # Dict-style outputs are also accepted.
    assert float(compute_loss({"logits": logits}, labels)) == pytest.approx(
        float(expected)
    )


def test_weighted_loss_requires_labels():
    compute_loss = weighted_cross_entropy(torch.ones(4))
    logits = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="labels=None"):
        compute_loss(SimpleNamespace(logits=logits), None)


def test_weighted_loss_detaches_weights_but_grad_flows_from_logits():
    weights = torch.tensor([1.0, 1.0, 1.0, 1.0], requires_grad=True)
    compute_loss = weighted_cross_entropy(weights)
    loss = compute_loss(
        SimpleNamespace(logits=torch.zeros(1, 4, requires_grad=True)),
        torch.tensor([1]),
    )
    assert loss.requires_grad
