"""Data helpers for text distillation experiments."""

from text_distillation.data.datasets import (
    TextClassificationDatasetInfo,
    get_dataset_info,
    get_train_eval_splits,
    list_supported_datasets,
    load_ag_news,
    load_text_classification_dataset,
    make_tiny_subset,
)

__all__ = [
    "TextClassificationDatasetInfo",
    "get_dataset_info",
    "get_train_eval_splits",
    "list_supported_datasets",
    "load_ag_news",
    "load_text_classification_dataset",
    "make_tiny_subset",
]
