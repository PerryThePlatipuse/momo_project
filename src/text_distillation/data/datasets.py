from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def load_ag_news(cache_dir: str | None = None):
    """Load the AG News DatasetDict from Hugging Face datasets."""
    from datasets import load_dataset

    return load_dataset("ag_news", cache_dir=cache_dir)


def get_label_names(dataset: Any, label_column: str = "label") -> list[str] | None:
    feature = getattr(dataset, "features", {}).get(label_column)
    names = getattr(feature, "names", None)
    return list(names) if names is not None else None


def make_tiny_subset(
    dataset: Any,
    n_per_class: int | None = None,
    total_size: int | None = None,
    label_column: str = "label",
    seed: int = 42,
):
    """Create a deterministic tiny subset for smoke checks.

    Use `n_per_class` for stratified sampling. Use `total_size` for a plain
    random subset. Exactly one of them must be provided.
    """
    if (n_per_class is None) == (total_size is None):
        raise ValueError("Provide exactly one of n_per_class or total_size.")

    rng = np.random.default_rng(seed)

    if total_size is not None:
        if total_size > len(dataset):
            raise ValueError(f"total_size={total_size} exceeds dataset size {len(dataset)}")
        indices = rng.choice(len(dataset), size=total_size, replace=False).tolist()
        return dataset.select(indices)

    indices_by_label: dict[Any, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset[label_column]):
        indices_by_label[label].append(index)

    selected: list[int] = []
    for label in sorted(indices_by_label):
        label_indices = indices_by_label[label]
        if n_per_class > len(label_indices):
            raise ValueError(
                f"n_per_class={n_per_class} exceeds class {label} size {len(label_indices)}"
            )
        selected.extend(rng.choice(label_indices, size=n_per_class, replace=False).tolist())

    return dataset.select(selected)

