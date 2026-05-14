from __future__ import annotations

import os
from itertools import islice
from pathlib import Path
from typing import Any, Literal

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
    max_steps: int | None = None,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    train_batch_size: int = 64,
    eval_batch_size: int = 128,
    gradient_accumulation_steps: int = 1,
    mixed_precision: Literal["auto", "fp16", "bf16", "no"] = "auto",
    scheduler_type: Literal["linear", "cosine"] = "linear",
    warmup_ratio: float = 0.1,
    max_grad_norm: float | None = 1.0,
    dataloader_num_workers: int = 2,
    seed: int = 42,
    save_model: bool = True,
    device: str | None = None,
    tracker: TimingTracker | None = None,
):
    """Fine-tune a pre-loaded sequence classifier with a small PyTorch loop.

    Two training-length modes:
    - default: train for `num_train_epochs` full passes.
    - paper-eval: pass `max_steps=N` to train for exactly N optimizer updates
      regardless of dataset size (data loader is cycled). This matches the
      DiLM-main `configs/test/coreset.yaml` evaluator protocol used for the
      Random / K-centers / Herding / DiLM rows of EXPECTED_METRICS.md.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    owns_tracker = tracker is None
    if tracker is None:
        tracker = TimingTracker()

    set_seed(seed)
    output_dir = ensure_dir(output_dir)
    device = device or get_device()

    # Resolve mixed precision. "auto" prefers fp16 on CUDA; bf16/fp16 are explicit.
    requested_dtype = _resolve_amp_dtype(mixed_precision, device, model)
    use_amp = requested_dtype is not None
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
        model_input_columns = _model_input_columns(tokenized_train, tokenizer)
        tokenized_train.set_format(type="torch", columns=model_input_columns)
        tokenized_eval.set_format(type="torch", columns=model_input_columns)

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

    if max_steps is not None:
        if max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        total_steps = int(max_steps)
    else:
        num_epochs = int(num_train_epochs)
        if num_epochs != num_train_epochs:
            raise ValueError("num_train_epochs must be an integer when max_steps is not set.")
        updates_per_epoch = max(1, int(np.ceil(len(train_loader) / gradient_accumulation_steps)))
        total_steps = max(1, updates_per_epoch * num_epochs)

    num_warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler_factory = {
        "linear": get_linear_schedule_with_warmup,
        "cosine": get_cosine_schedule_with_warmup,
    }[scheduler_type]
    scheduler = scheduler_factory(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_steps,
    )

    # GradScaler is fp16-only — bf16 does not need scaling.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and requested_dtype == torch.float16)
    running_losses: list[float] = []
    model.train()

    with tracker.measure("training_sec"):
        if max_steps is not None:
            _train_for_steps(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                amp_dtype=requested_dtype,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                total_steps=total_steps,
                running_losses=running_losses,
            )
        else:
            _train_for_epochs(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
                use_amp=use_amp,
                amp_dtype=requested_dtype,
                gradient_accumulation_steps=gradient_accumulation_steps,
                max_grad_norm=max_grad_norm,
                num_epochs=num_epochs,
                running_losses=running_losses,
            )

    with tracker.measure("evaluation_sec"):
        eval_metrics = _evaluate_model(
            model,
            eval_loader,
            device,
            use_amp=use_amp,
            amp_dtype=requested_dtype,
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


def _resolve_amp_dtype(mixed_precision: str, device: str, model: Any):
    import torch

    if mixed_precision == "no" or device != "cuda":
        return None
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    if mixed_precision == "auto":
        # DeBERTa v3 (config.model_type == "deberta-v2") leaves some tensors in fp16
        # after from_pretrained, which makes GradScaler.unscale_ raise
        # "Attempting to unscale FP16 gradients." Skip AMP for it on "auto".
        if getattr(model.config, "model_type", None) == "deberta-v2":
            print(f"[train_text_classifier] disabling fp16 for {model.config.model_type} (known incompatibility with GradScaler)")
            return None
        return torch.float16
    raise ValueError(f"mixed_precision must be one of: auto, fp16, bf16, no — got {mixed_precision!r}")


def _model_input_columns(tokenized_dataset: Any, tokenizer: Any) -> list[str]:
    candidate_columns = [*getattr(tokenizer, "model_input_names", ()), "labels"]
    return [column for column in candidate_columns if column in tokenized_dataset.column_names]


def _train_for_epochs(*, model, loader, optimizer, scheduler, scaler, device, use_amp,
                      amp_dtype, gradient_accumulation_steps, max_grad_norm,
                      num_epochs, running_losses):
    import torch

    for epoch in range(num_epochs):
        progress = tqdm(loader, desc=f"Training epoch {epoch + 1}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(progress):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
            loss = outputs.loss
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()

            should_update = (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(loader)
            if should_update:
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            loss_value = float(loss.detach().cpu())
            running_losses.append(loss_value)
            progress.set_postfix(loss=loss_value)


def _train_for_steps(*, model, loader, optimizer, scheduler, scaler, device, use_amp,
                     amp_dtype, gradient_accumulation_steps, max_grad_norm,
                     total_steps, running_losses):
    """Fixed-`max_steps` loop — cycles the dataloader for very small distilled subsets."""
    import torch

    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(range(total_steps), desc=f"Training {total_steps} steps")
    completed = 0
    while completed < total_steps:
        for accum_step, batch in enumerate(islice(_cycle(loader), gradient_accumulation_steps)):
            batch = move_batch_to_device(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(**batch)
            loss = outputs.loss
            scaled_loss = loss / gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
            loss_value = float(loss.detach().cpu())
            running_losses.append(loss_value)

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        completed += 1
        progress.update(1)
        progress.set_postfix(loss=running_losses[-1])


def _cycle(iterable):
    while True:
        for item in iterable:
            yield item


def _evaluate_model(
    model: Any,
    dataloader: Any,
    device: str,
    use_amp: bool = False,
    amp_dtype: Any = None,
    metric_name: str | None = None,
) -> dict[str, float]:
    import torch

    model.eval()
    predictions: list[int] = []
    labels: list[int] = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = move_batch_to_device(batch, device)
            batch_labels = batch["labels"]
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
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
