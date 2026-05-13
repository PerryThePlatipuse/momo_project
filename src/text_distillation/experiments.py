"""Helpers for the explicit baseline pipeline: load → select → train → save.

Notebooks own the experiment flow. These two helpers exist purely to remove
boilerplate at the ends of that flow:

- `load_baseline_data` packages the standard dataset loading sequence
  (`get_dataset_info` → `load_text_classification_dataset` → splits →
  optional sub-sampling → label names → num_labels) into one call.
- `save_baseline_run` atomically writes `config.json`, `metrics.json` and
  (for distillation methods) `distilled_dataset/` under one run directory.

Selection and training stay in the notebook so the pipeline is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from text_distillation.data.datasets import (
    TextClassificationDatasetInfo,
    get_dataset_info,
    get_label_names,
    get_train_eval_splits,
    load_text_classification_dataset,
    make_tiny_subset,
)
from text_distillation.model.registry import get_model_profile
from text_distillation.saving import (
    save_distilled_dataset,
    save_experiment_config,
    save_metrics,
)
from text_distillation.timing import TimingTracker
from text_distillation.utils import get_git_commit_hash


@dataclass(frozen=True)
class BaselineData:
    """All inputs a baseline notebook needs after the data step."""

    dataset_info: TextClassificationDatasetInfo
    train_pool: Any
    eval_dataset: Any
    label_names: list[str] | None
    num_labels: int
    full_train_size: int
    train_pool_size: int


def load_baseline_data(
    dataset_name: str,
    *,
    seed: int = 42,
    max_train_pool_dpc: int | None = None,
    max_eval_samples: int | None = None,
) -> BaselineData:
    """Load a supported dataset, build train/eval splits and label metadata.

    `max_train_pool_dpc` restricts the train pool to N examples per class
    (smoke-check helper). `max_eval_samples` does the same on the eval split
    via plain random sub-sampling.
    """
    dataset_info = get_dataset_info(dataset_name)
    dataset = load_text_classification_dataset(dataset_name)
    full_train, eval_dataset = get_train_eval_splits(dataset, dataset_name)
    full_train_size = len(full_train)

    train_pool = full_train
    if max_train_pool_dpc is not None:
        train_pool = make_tiny_subset(
            full_train,
            n_per_class=max_train_pool_dpc,
            label_column=dataset_info.label_column,
            seed=seed,
        )
    if max_eval_samples is not None:
        eval_dataset = make_tiny_subset(eval_dataset, total_size=max_eval_samples, seed=seed)

    label_names = get_label_names(full_train, dataset_info.label_column)
    num_labels = len(label_names) if label_names else len(set(full_train[dataset_info.label_column]))

    return BaselineData(
        dataset_info=dataset_info,
        train_pool=train_pool,
        eval_dataset=eval_dataset,
        label_names=label_names,
        num_labels=num_labels,
        full_train_size=full_train_size,
        train_pool_size=len(train_pool),
    )


def save_baseline_run(
    run_dir: str | Path,
    *,
    data: BaselineData,
    method_name: str,
    model_name: str,
    k_per_class: int | None,
    seed: int,
    train_dataset: Any,
    metrics: dict[str, Any],
    tracker: TimingTracker | None = None,
    project_root: str | Path | None = None,
    extra_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist a single baseline run: config.json + metrics.json + distilled dataset.

    `train_dataset` is the dataset that was actually used for training. For
    distillation methods it is the selected subset (saved separately). For
    `full_data` it is the train pool itself and no separate copy is saved.

    Returns a flat dict of ``{**config, **metrics}`` for in-notebook tally.
    """
    run_dir = Path(run_dir)
    train_size = len(train_dataset)
    profile = get_model_profile(model_name)
    config: dict[str, Any] = {
        "experiment_name": run_dir.name,
        "dataset_name": data.dataset_info.name,
        "method_name": method_name,
        "model_name": model_name,
        "model_family": profile.family,
        "dpc": k_per_class,
        "k_total": train_size,
        "text_columns": list(data.dataset_info.text_columns),
        "label_column": data.dataset_info.label_column,
        "metric_name": data.dataset_info.metric_name,
        "train_split": data.dataset_info.train_split,
        "eval_split": data.dataset_info.eval_split,
        "seed": seed,
        "full_train_size": data.full_train_size,
        "train_pool_size": data.train_pool_size,
        "train_size": train_size,
        "eval_size": len(data.eval_dataset),
        "compression_ratio": data.full_train_size / train_size if train_size else None,
        "git_commit": get_git_commit_hash(project_root or "."),
    }
    if extra_config:
        config.update(extra_config)
    save_experiment_config(config, run_dir)
    if method_name != "full_data":
        save_distilled_dataset(train_dataset, run_dir)

    if tracker is not None and "timings" not in metrics:
        metrics = {**metrics, "timings": tracker.as_dict()}
    save_metrics(metrics, run_dir)
    return {**config, **metrics}
