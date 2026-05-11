from __future__ import annotations

from pathlib import Path
from typing import Any

from text_distillation.utils import ensure_dir, save_json


def create_run_dir(base_dir: str | Path, experiment_name: str) -> Path:
    return ensure_dir(Path(base_dir) / experiment_name)


def save_experiment_config(config: dict[str, Any], run_dir: str | Path) -> Path:
    return save_json(config, Path(run_dir) / "config.json")


def save_metrics(metrics: dict[str, Any], run_dir: str | Path) -> Path:
    return save_json(metrics, Path(run_dir) / "metrics.json")


def save_predictions(predictions: list[dict[str, Any]], run_dir: str | Path) -> Path:
    return save_json({"predictions": predictions}, Path(run_dir) / "predictions.json")


def save_distilled_dataset(dataset: Any, run_dir: str | Path, name: str = "distilled_dataset") -> Path:
    output_dir = ensure_dir(Path(run_dir) / name)
    dataset.save_to_disk(str(output_dir))
    return output_dir

