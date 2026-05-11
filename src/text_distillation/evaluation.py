from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from text_distillation.utils import get_device, move_batch_to_device


def compute_accuracy(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def compute_f1_macro(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def compute_f1_binary(y_true: list[int] | np.ndarray, y_pred: list[int] | np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="binary"))


def compute_accuracy_f1_average(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
) -> float:
    return (compute_accuracy(y_true, y_pred) + compute_f1_binary(y_true, y_pred)) / 2.0


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    metric_name: str | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": compute_accuracy(y_true, y_pred),
        "f1_macro": compute_f1_macro(y_true, y_pred),
    }
    if metric_name == "accuracy":
        metrics["score"] = metrics["accuracy"]
    elif metric_name == "accuracy_f1_average":
        metrics["f1"] = compute_f1_binary(y_true, y_pred)
        metrics["score"] = (metrics["accuracy"] + metrics["f1"]) / 2.0
    elif metric_name is not None:
        raise ValueError(f"Unsupported metric_name: {metric_name}")
    return metrics


def evaluate_classifier(
    model: Any,
    dataloader: Any,
    device: str | None = None,
    metric_name: str | None = None,
) -> dict[str, Any]:
    """Evaluate a PyTorch classifier returning metrics and predictions."""
    import torch

    device = device or get_device()
    model.to(device)
    model.eval()

    predictions: list[int] = []
    labels: list[int] = []

    with torch.inference_mode():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            batch_labels = batch.pop("labels")
            outputs = model(**batch)
            batch_predictions = outputs.logits.argmax(dim=-1)
            predictions.extend(batch_predictions.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())

    metrics = compute_classification_metrics(labels, predictions, metric_name=metric_name)
    metrics["predictions"] = predictions
    metrics["labels"] = labels
    return metrics
