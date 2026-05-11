from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TextClassificationDatasetInfo:
    name: str
    hf_path: str
    hf_config: str | None
    train_split: str
    eval_split: str
    test_split: str | None
    text_columns: tuple[str, ...]
    label_column: str
    metric_name: str


SUPPORTED_DATASETS: dict[str, TextClassificationDatasetInfo] = {
    "ag_news": TextClassificationDatasetInfo(
        name="ag_news",
        hf_path="ag_news",
        hf_config=None,
        train_split="train",
        eval_split="test",
        test_split="test",
        text_columns=("text",),
        label_column="label",
        metric_name="accuracy",
    ),
    "sst2": TextClassificationDatasetInfo(
        name="sst2",
        hf_path="glue",
        hf_config="sst2",
        train_split="train",
        eval_split="validation",
        test_split="test",
        text_columns=("sentence",),
        label_column="label",
        metric_name="accuracy",
    ),
    "qqp": TextClassificationDatasetInfo(
        name="qqp",
        hf_path="glue",
        hf_config="qqp",
        train_split="train",
        eval_split="validation",
        test_split="test",
        text_columns=("question1", "question2"),
        label_column="label",
        metric_name="accuracy_f1_average",
    ),
    "mnli-m": TextClassificationDatasetInfo(
        name="mnli-m",
        hf_path="glue",
        hf_config="mnli",
        train_split="train",
        eval_split="validation_matched",
        test_split="test_matched",
        text_columns=("premise", "hypothesis"),
        label_column="label",
        metric_name="accuracy",
    ),
    "mnli": TextClassificationDatasetInfo(
        name="mnli",
        hf_path="glue",
        hf_config="mnli",
        train_split="train",
        eval_split="validation_matched",
        test_split="test_matched",
        text_columns=("premise", "hypothesis"),
        label_column="label",
        metric_name="accuracy",
    ),
}


def list_supported_datasets() -> list[str]:
    return sorted(SUPPORTED_DATASETS)


def get_dataset_info(dataset_name: str) -> TextClassificationDatasetInfo:
    key = dataset_name.lower().replace("_", "-")
    if key == "ag-news":
        key = "ag_news"
    if key == "mnli_matched":
        key = "mnli-m"
    if key not in SUPPORTED_DATASETS:
        supported = ", ".join(list_supported_datasets())
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported datasets: {supported}")
    return SUPPORTED_DATASETS[key]


def load_ag_news(cache_dir: str | None = None):
    """Load the AG News DatasetDict from Hugging Face datasets."""
    from datasets import load_dataset

    return load_dataset("ag_news", cache_dir=cache_dir)


def load_text_classification_dataset(dataset_name: str, cache_dir: str | None = None):
    """Load a supported text classification dataset by project name."""
    from datasets import load_dataset

    info = get_dataset_info(dataset_name)
    if info.hf_config is None:
        return load_dataset(info.hf_path, cache_dir=cache_dir)
    return load_dataset(info.hf_path, info.hf_config, cache_dir=cache_dir)


def get_train_eval_splits(dataset_dict: Any, dataset_name: str):
    """Return `(train_dataset, eval_dataset)` for a supported dataset."""
    info = get_dataset_info(dataset_name)
    return dataset_dict[info.train_split], dataset_dict[info.eval_split]


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
