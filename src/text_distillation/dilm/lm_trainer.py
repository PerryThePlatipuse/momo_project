"""LM-phase trainer — standard causal-LM fine-tuning of the class-conditional generator.

Same loss as `DiLM-main/src/distillation/trainer_lm.py`: per-sample CE on
`<bos_y> sent_a <sep> sent_b`. The trained generator is the starting point
for either Vanilla LM (sample immediately) or DiLM (continue with DC trainer).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from text_distillation.timing import TimingTracker

from .generator import GeneratorModel


def train_generator_lm(
    generator: GeneratorModel,
    dataset: Any,
    label_column: str,
    *,
    max_steps: int = 5_000,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float | None = 1.0,
    seed: int = 42,
    max_finetune_samples: int | None = 8_000,
    tracker: TimingTracker | None = None,
) -> None:
    """Train `generator` in-place with causal-LM loss for `max_steps` updates.

    `max_steps` defaults to 5k (a few minutes on a single GPU). The DiLM paper
    runs 80k; bump it up if you have the compute and want paper-faithful numbers.
    """
    generator.train()
    device = next(generator.model.parameters()).device

    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if max_finetune_samples is not None and len(dataset) > max_finetune_samples:
        indices = rng.choice(len(dataset), size=max_finetune_samples, replace=False)

    texts: list[str] = []
    for idx in indices:
        example = dataset[int(idx)]
        texts.append(generator.build_training_text(example, int(example[label_column])))

    max_length = generator.config.generate_max_length
    encodings = generator.encode_training_batch(texts, max_length=max_length)

    class _IndexDS(torch.utils.data.Dataset):
        def __len__(self_inner) -> int:
            return len(texts)

        def __getitem__(self_inner, i):
            return {k: v[i] for k, v in encodings.items()}

    loader = DataLoader(_IndexDS(), batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = AdamW(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    _measure = tracker.measure if tracker is not None else _noop_context

    with _measure("dilm_lm_train_sec"):
        step = 0
        pbar = tqdm(total=max_steps, desc="DiLM LM phase", leave=False)
        while step < max_steps:
            for batch in loader:
                if step >= max_steps:
                    break
                batch = {k: v.to(device) for k, v in batch.items()}
                losses = generator.compute_loss(**batch)
                loss = losses.mean()

                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                step += 1
                pbar.update(1)
                pbar.set_postfix(loss=float(loss.detach().cpu()))
        pbar.close()


class _noop_context:
    def __init__(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
