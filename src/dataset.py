"""Torch dataset wrapper: pandas frame -> tokenised model inputs."""

from __future__ import annotations

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from src.schema import LABEL_COLUMN, TEXT_COLUMN, to_model_label


class SeverityDataset(Dataset):
    """Yields model-space labels (0..3), not domain levels (1..4)."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 256,
    ) -> None:
        self.texts = frame[TEXT_COLUMN].tolist()
        levels = frame[LABEL_COLUMN].to_numpy().astype(int)
        self.labels = torch.tensor(
            to_model_label(levels), dtype=torch.long
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item = {k: torch.tensor(v, dtype=torch.long) for k, v in enc.items()}
        item["labels"] = self.labels[idx]
        return item
