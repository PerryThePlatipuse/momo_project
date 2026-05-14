"""Vanilla LM distillation — class-conditional language model baseline from DiLM.

Direct adaptation of `DiLM-main/src/generator.py` + `DiLM-main/src/distillation/trainer_lm.py`,
collapsed into one function that fits this project's notebook flow.

Approach (same as DiLM paper, simplified loop):
1. Build one shared generator (`distilgpt2` by default) with:
   - `<pad>` for batching;
   - `<sep>` for splitting sentence pairs (QQP, MNLI);
   - per-label `<bos_0>`, `<bos_1>`, ... initialized from the model's BOS embedding.
2. Fine-tune the generator on real training text with standard causal-LM loss.
   The text fed in is built per-example as ``<bos_y> sent_a <sep> sent_b``, so
   the generator learns to condition on the label token.
3. Sample `k_per_class` continuations starting from each `<bos_y>`, decode
   back into one or two text fields, return a `datasets.Dataset`.

This is the baseline DiLM beats. Use as a drop-in `select_*` replacement.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from datasets import Dataset

from text_distillation.data.transforms import TextColumns, normalize_text_columns
from text_distillation.dilm.official import (
    PAPER_N_DATASETS,
    load_paper_vanilla_lm_dataset,
    run_paper_vanilla_lm_reproduction,
)
from text_distillation.distillation import register_selection
from text_distillation.timing import TimingTracker
from text_distillation.utils import get_device, set_seed


SEP_TOKEN = "<sep>"


@register_selection("vanilla_lm")
def distill_vanilla_lm(
    dataset: Any,
    dataset_name: str | None = None,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    label_column: str = "label",
    k_per_class: int = 20,
    seed: int = 42,
    generator_model_name: str = "distilgpt2",
    num_train_epochs: int = 3,
    train_batch_size: int = 8,
    learning_rate: float = 5e-5,
    max_length: int = 128,
    generate_batch_size: int = 32,
    top_p: float = 0.95,
    temperature: float = 1.0,
    max_finetune_samples: int | None = 4000,
    device: str | None = None,
    tracker: TimingTracker | None = None,
    dataset_index: int = 0,
    output_root: str = "artifacts/vanilla_lm_paper",
    data_root: str = "data/vanilla_lm_paper",
    n_datasets: int = PAPER_N_DATASETS,
    force: bool = False,
) -> Any:
    """Train a class-conditional LM on `dataset` and sample `k_per_class` per class.

    Returns a `datasets.Dataset` whose schema matches the input (same text columns
    + `label_column`). Drop-in replacement for any `select_*` coreset function.
    """
    if dataset_name is not None:
        run_paper_vanilla_lm_reproduction(
            dataset_name,
            dpc=k_per_class,
            seed=seed,
            output_root=output_root,
            data_root=data_root,
            n_datasets=n_datasets,
            force=force,
        )
        return load_paper_vanilla_lm_dataset(
            dataset_name,
            dpc=k_per_class,
            seed=seed,
            dataset_index=dataset_index,
            output_root=output_root,
            data_root=data_root,
        )

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from torch.optim import AdamW
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    set_seed(seed)
    device = device or get_device()
    columns = normalize_text_columns(text_column=text_column, text_columns=text_columns)
    labels = sorted(set(dataset[label_column]))
    num_labels = len(labels)

    # --- 1. Build the generator + label-aware tokenizer (DiLM-main/generator.py) ---
    tokenizer = AutoTokenizer.from_pretrained(generator_model_name)
    model = AutoModelForCausalLM.from_pretrained(generator_model_name)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
    if SEP_TOKEN not in tokenizer.get_vocab():
        tokenizer.add_tokens(SEP_TOKEN)
    sep_token_id = tokenizer.convert_tokens_to_ids(SEP_TOKEN)

    bos_tokens = {label: f"<bos_{label}>" for label in labels}
    bos_ids: dict[Any, int] = {}
    for label, bos in bos_tokens.items():
        if bos not in tokenizer.get_vocab():
            tokenizer.add_tokens(bos)
        bos_ids[label] = tokenizer.convert_tokens_to_ids(bos)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize each <bos_y> embedding from the model's original BOS so it
    # starts somewhere sensible instead of random.
    init_bos_id = tokenizer.bos_token_id or tokenizer.eos_token_id
    with torch.no_grad():
        ref = model.get_input_embeddings().weight[init_bos_id].clone()
        for bos_id in bos_ids.values():
            model.get_input_embeddings().weight[bos_id].copy_(ref)

    # --- 2. Build training texts: "<bos_y> sentence_a <sep> sentence_b" ---
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    if max_finetune_samples is not None and len(dataset) > max_finetune_samples:
        indices = rng.choice(len(dataset), size=max_finetune_samples, replace=False)

    train_texts: list[str] = []
    for idx in indices:
        example = dataset[int(idx)]
        bos = bos_tokens[example[label_column]]
        parts = [str(example[col]) for col in columns]
        train_texts.append(bos + " " + f" {SEP_TOKEN} ".join(parts))

    encodings = tokenizer(
        train_texts,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # --- 3. Fine-tune (single-pass standard CLM loss, DiLM's trainer_lm.py) ---
    model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = max(1, num_train_epochs * ((len(train_texts) + train_batch_size - 1) // train_batch_size))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    _measure = tracker.measure if tracker is not None else _noop_context

    with _measure("vanilla_lm_finetune_sec"):
        for _epoch in range(num_train_epochs):
            perm = torch.randperm(len(train_texts))
            for batch_start in range(0, len(train_texts), train_batch_size):
                batch_idx = perm[batch_start : batch_start + train_batch_size]
                batch_ids = input_ids[batch_idx].to(device)
                batch_mask = attention_mask[batch_idx].to(device)
                labels_ids = batch_ids.clone()
                labels_ids[batch_mask == 0] = -100

                outputs = model(
                    input_ids=batch_ids,
                    attention_mask=batch_mask,
                    labels=labels_ids,
                )
                optimizer.zero_grad()
                outputs.loss.backward()
                optimizer.step()
                scheduler.step()

    # --- 4. Sample k_per_class continuations per label, decode back to fields ---
    output: dict[str, list] = {col: [] for col in columns}
    output[label_column] = []

    model.eval()
    bad_words = [[bos_id] for bos_id in bos_ids.values()]  # forbid generating extra BOS tokens

    with _measure("vanilla_lm_sample_sec"):
        for label in labels:
            prompt = torch.tensor([[bos_ids[label]]] * generate_batch_size, device=device)
            remaining = k_per_class
            attempts = 0
            while remaining > 0 and attempts < k_per_class * 4:
                batch_n = min(remaining, generate_batch_size)
                batch_prompt = prompt[:batch_n]
                with torch.inference_mode():
                    gen = model.generate(
                        batch_prompt,
                        do_sample=True,
                        top_p=top_p,
                        temperature=temperature,
                        max_length=max_length,
                        pad_token_id=tokenizer.pad_token_id,
                        bad_words_ids=bad_words,
                    )
                # Strip the leading BOS, decode.
                texts = tokenizer.batch_decode(gen[:, 1:], skip_special_tokens=True)
                for text in texts:
                    if remaining == 0:
                        break
                    parts = [p.strip() for p in text.split(SEP_TOKEN, maxsplit=len(columns) - 1)]
                    if len(parts) < len(columns):
                        attempts += 1
                        continue
                    for col, part in zip(columns, parts):
                        output[col].append(part or " ")
                    output[label_column].append(label)
                    remaining -= 1
                attempts += batch_n

            # If sampling failed (rare on tiny models), fall back to empty filler.
            while remaining > 0:
                for col in columns:
                    output[col].append(" ")
                output[label_column].append(label)
                remaining -= 1

    return Dataset.from_dict(output)


class _noop_context:
    def __init__(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
