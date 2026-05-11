from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from text_distillation.data.dataloaders import create_text_dataloader
from text_distillation.data.transforms import tokenize_text_dataset
from text_distillation.evaluation import compute_classification_metrics
from text_distillation.model.loading import load_sequence_classifier, load_tokenizer
from text_distillation.utils import ensure_dir, get_device, move_batch_to_device, set_seed


def train_text_classifier(
    train_dataset: Any,
    eval_dataset: Any,
    model_name: str,
    output_dir: str | Path,
    text_column: str = "text",
    label_column: str = "label",
    num_labels: int | None = None,
    label_names: list[str] | None = None,
    max_length: int = 128,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    seed: int = 42,
    save_model: bool = True,
    device: str | None = None,
):
    """Fine-tune a sequence classifier with a small PyTorch loop.

    Returns `(model, metrics)` so notebooks can inspect the model when needed
    while still getting a simple metrics dictionary for saving.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    set_seed(seed)
    output_dir = ensure_dir(output_dir)
    device = device or get_device()

    if num_labels is None:
        num_labels = len(set(train_dataset[label_column]))

    tokenizer = load_tokenizer(model_name)
    model = load_sequence_classifier(
        model_name=model_name,
        num_labels=num_labels,
        label_names=label_names,
    )

    tokenized_train = tokenize_text_dataset(
        train_dataset,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    tokenized_eval = tokenize_text_dataset(
        eval_dataset,
        tokenizer=tokenizer,
        text_column=text_column,
        label_column=label_column,
        max_length=max_length,
    )
    tokenized_train.set_format(type="torch")
    tokenized_eval.set_format(type="torch")

    train_loader = create_text_dataloader(
        tokenized_train,
        tokenizer=tokenizer,
        batch_size=train_batch_size,
        shuffle=True,
    )
    eval_loader = create_text_dataloader(
        tokenized_eval,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        shuffle=False,
    )

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = int(num_train_epochs)
    if num_epochs != num_train_epochs:
        raise ValueError("num_train_epochs must be an integer value for the simple training loop.")

    total_steps = max(1, len(train_loader) * num_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    running_losses: list[float] = []
    model.train()
    for epoch in range(num_epochs):
        progress = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{num_epochs}")
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_value = float(loss.detach().cpu())
            running_losses.append(loss_value)
            progress.set_postfix(loss=loss_value)

    eval_metrics = _evaluate_model(model, eval_loader, device)

    metrics = {
        "train_loss": float(np.mean(running_losses)) if running_losses else 0.0,
        **{key: _to_float(value) for key, value in eval_metrics.items()},
    }

    if save_model:
        model.save_pretrained(str(output_dir / "model"))
        tokenizer.save_pretrained(str(output_dir / "model"))

    return model, metrics


def _evaluate_model(model: Any, dataloader: Any, device: str) -> dict[str, float]:
    import torch

    model.eval()
    predictions: list[int] = []
    labels: list[int] = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            batch_labels = batch["labels"]
            outputs = model(**batch)
            batch_predictions = outputs.logits.argmax(dim=-1)
            predictions.extend(batch_predictions.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())

    model.train()
    return compute_classification_metrics(labels, predictions)


def _to_float(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value
