from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from text_distillation.data.dataloaders import create_text_dataloader
from text_distillation.data.transforms import TextColumns, tokenize_text_dataset
from text_distillation.evaluation import compute_classification_metrics
from text_distillation.timing import TimingTracker
from text_distillation.utils import ensure_dir, get_device, move_batch_to_device, set_seed


def train_text_classifier(
    *,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    eval_dataset: Any,
    output_dir: str | Path,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    label_column: str = "label",
    max_length: int = 128,
    metric_name: str | None = None,
    num_train_epochs: float = 3.0,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    train_batch_size: int = 64,
    eval_batch_size: int = 128,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "auto",
    dataloader_num_workers: int = 2,
    seed: int = 42,
    save_model: bool = True,
    device: str | None = None,
    tracker: TimingTracker | None = None,
):
    """Fine-tune a pre-loaded sequence classifier with a small PyTorch loop.

    The caller is responsible for constructing `model` and `tokenizer` (see
    `load_sequence_classifier` / `load_tokenizer`) — making the model step
    visible in the notebook. Returns `(model, metrics)`.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    owns_tracker = tracker is None
    if tracker is None:
        tracker = TimingTracker()

    set_seed(seed)
    output_dir = ensure_dir(output_dir)
    device = device or get_device()
    use_amp = mixed_precision == "fp16" or (mixed_precision == "auto" and device == "cuda")
    # DeBERTa v3 (config.model_type == "deberta-v2") leaves some tensors in fp16
    # after from_pretrained, which makes torch.amp.GradScaler.unscale_ raise
    # "Attempting to unscale FP16 gradients." Disable AMP automatically.
    if use_amp and mixed_precision == "auto" and getattr(model.config, "model_type", None) == "deberta-v2":
        print(f"[train_text_classifier] disabling fp16 for {model.config.model_type} (known incompatibility with GradScaler)")
        use_amp = False
    pin_memory = device == "cuda"

    with tracker.measure("tokenization_sec"):
        tokenized_train = tokenize_text_dataset(
            train_dataset,
            tokenizer=tokenizer,
            text_column=text_column,
            text_columns=text_columns,
            label_column=label_column,
            max_length=max_length,
        )
        tokenized_eval = tokenize_text_dataset(
            eval_dataset,
            tokenizer=tokenizer,
            text_column=text_column,
            text_columns=text_columns,
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
        num_workers=dataloader_num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = create_text_dataloader(
        tokenized_eval,
        tokenizer=tokenizer,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=dataloader_num_workers,
        pin_memory=pin_memory,
    )

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    num_epochs = int(num_train_epochs)
    if num_epochs != num_train_epochs:
        raise ValueError("num_train_epochs must be an integer value for the simple training loop.")

    updates_per_epoch = int(np.ceil(len(train_loader) / gradient_accumulation_steps))
    total_steps = max(1, updates_per_epoch * num_epochs)
    # 10% warmup. RoBERTa-family in particular often gets stuck predicting the
    # majority class without it, especially on small distilled subsets.
    num_warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    running_losses: list[float] = []
    model.train()
    with tracker.measure("training_sec"):
        for epoch in range(num_epochs):
            progress = tqdm(train_loader, desc=f"Training epoch {epoch + 1}/{num_epochs}")
            optimizer.zero_grad(set_to_none=True)
            for step, batch in enumerate(progress):
                batch = move_batch_to_device(batch, device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                    outputs = model(**batch)
                loss = outputs.loss
                scaled_loss = loss / gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

                should_update = (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(train_loader)
                if should_update:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                loss_value = float(loss.detach().cpu())
                running_losses.append(loss_value)
                progress.set_postfix(loss=loss_value)

    with tracker.measure("evaluation_sec"):
        eval_metrics = _evaluate_model(
            model,
            eval_loader,
            device,
            mixed_precision=mixed_precision,
            metric_name=metric_name,
        )

    metrics = {
        "train_loss": float(np.mean(running_losses)) if running_losses else 0.0,
        **{key: _to_float(value) for key, value in eval_metrics.items()},
    }
    if owns_tracker:
        metrics["timings"] = tracker.as_dict()

    if save_model:
        model.save_pretrained(str(output_dir / "model"))
        tokenizer.save_pretrained(str(output_dir / "model"))

    return model, metrics


def _evaluate_model(
    model: Any,
    dataloader: Any,
    device: str,
    mixed_precision: str = "auto",
    metric_name: str | None = None,
) -> dict[str, float]:
    import torch

    model.eval()
    use_amp = mixed_precision == "fp16" or (mixed_precision == "auto" and device == "cuda")
    predictions: list[int] = []
    labels: list[int] = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            batch_labels = batch["labels"]
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model(**batch)
            batch_predictions = outputs.logits.argmax(dim=-1)
            predictions.extend(batch_predictions.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())

    model.train()
    return compute_classification_metrics(labels, predictions, metric_name=metric_name)


def _to_float(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return float(value)
    return value
