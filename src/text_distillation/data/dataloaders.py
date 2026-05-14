from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader


def collate_text_features(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack tokenized text features into a PyTorch batch."""
    features = [dict(feature) for feature in features]
    batch: dict[str, Any] = {}
    for key in features[0]:
        values = [feature[key] for feature in features]
        if torch.is_tensor(values[0]):
            batch[key] = torch.stack(values)
        else:
            dtype = torch.long if key in {"input_ids", "attention_mask", "token_type_ids", "labels"} else None
            batch[key] = torch.tensor(values, dtype=dtype)
    return batch


def create_text_dataloader(
    dataset: Any,
    tokenizer: Any | None = None,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Create a DataLoader for tokenized text datasets."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_text_features,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
