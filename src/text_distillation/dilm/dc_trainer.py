"""DC-phase trainer — DiLM's gradient-matching fine-tuning of the generator.

Simplified port of `DiLM-main/src/distillation/trainer_dc.py`. We keep the core
DiLM idea — train the generator so its samples produce the same classifier
gradient as real data — but drop several productivity knobs that are not
needed for a working baseline:

- ``classifier_grad_only=True``: only the BERT classification head's params
  contribute to the matching loss. Without this the gradient vectors are too
  high-dimensional to align stably.
- One real-data batch per label per step (no ``gm_real_grad_accum_step``).
- Single inner step (no inner learner training between outer updates).
- No cluster-balanced sampling (``n_clusters_for_*``).
- No repset_teacher pre-computation — we sample fresh real batches each step.

The matching loss per label is ``1 - cosine_similarity(grad_real, grad_syn)``.
Per-sample LM losses on synthetic batches are softmax-weighted into
``loss_weights`` so the generator gradient flows through the dot product.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from text_distillation.timing import TimingTracker

from .generator import GeneratorModel


def train_generator_dc(
    generator: GeneratorModel,
    learner: Any,
    learner_tokenizer: Any,
    dataset: Any,
    label_column: str,
    *,
    max_steps: int = 2_000,
    gm_real_dpc: int = 64,
    gm_syn_dpc: int = 16,
    normalize_temperature: float = 1.0,
    learning_rate: float = 3e-7,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.05,
    max_grad_norm: float | None = 1.0,
    generate_interval: int = 20,
    generate_max_length: int | None = None,
    seed: int = 42,
    tracker: TimingTracker | None = None,
) -> None:
    """Fine-tune `generator` in-place with the DiLM gradient-matching loss.

    `learner` is a frozen BERT-family classifier whose head defines what
    "matching the gradient" means. `dataset` provides real text — we sample a
    fresh `gm_real_dpc`-sized batch per label per outer step.
    """
    generator.train()
    learner.eval()
    for p in learner.parameters():
        p.requires_grad_(False)
    device = next(generator.model.parameters()).device
    learner.to(device)

    num_labels = generator.num_labels
    indices_by_label = _indices_by_label(dataset, label_column)

    optimizer = AdamW(generator.parameters(), lr=learning_rate, weight_decay=weight_decay)
    warmup_steps = max(1, int(max_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, max_steps)

    classifier_param_names = _classifier_param_names(learner)
    # functional_call view of learner params (frozen).
    params = {k: v.detach() for k, v in learner.named_parameters()}
    buffers = {k: v.detach() for k, v in learner.named_buffers()}

    rng = torch.Generator().manual_seed(seed)
    _measure = tracker.measure if tracker is not None else _noop_context

    synthetic_per_label: dict[int, list[str]] | None = None

    with _measure("dilm_dc_train_sec"):
        pbar = tqdm(range(max_steps), desc="DiLM DC phase", leave=False)
        for step in pbar:
            # Refresh synthetic samples every `generate_interval` steps. Generation
            # is the expensive part — amortize it across many outer updates.
            if synthetic_per_label is None or step % generate_interval == 0:
                synthetic_per_label = {
                    label: generator.sample(
                        label,
                        n=gm_syn_dpc * generate_interval,
                        max_length=generate_max_length or generator.config.generate_max_length,
                    )
                    for label in range(num_labels)
                }

            step_loss = 0.0
            optimizer.zero_grad(set_to_none=True)
            for label in range(num_labels):
                # ---- real-data gradient on the classifier head (no_grad context) ----
                real_idx = _sample_indices(indices_by_label[label], gm_real_dpc, rng)
                real_text_batch = _gather_text(dataset, real_idx, generator.sentence_keys)
                real_inputs = _tokenize_for_learner(
                    learner_tokenizer, real_text_batch, label, num_labels,
                    max_length=generator.config.generate_max_length, device=device,
                )
                with torch.no_grad():
                    grad_real = _compute_grad(
                        learner, params, buffers, real_inputs,
                        classifier_param_names=classifier_param_names,
                    ).detach()

                # ---- synthetic-data gradient (with gradient flowing into generator) ----
                syn_texts = _pop_synthetic(synthetic_per_label[label], gm_syn_dpc)
                syn_inputs_gen = _tokenize_for_generator(
                    generator, syn_texts, label,
                    max_length=generator.config.generate_max_length, device=device,
                )
                # per-sample LM losses with grad_fn so the matching loss backprops
                # through the generator parameters.
                gen_losses = generator.compute_loss(**syn_inputs_gen)
                loss_weights = F.softmax(-gen_losses / normalize_temperature, dim=-1)

                syn_inputs_learner = _tokenize_for_learner(
                    learner_tokenizer, syn_texts, label, num_labels,
                    max_length=generator.config.generate_max_length, device=device,
                )
                grad_syn = _compute_grad(
                    learner, params, buffers, syn_inputs_learner,
                    loss_weights=loss_weights,
                    classifier_param_names=classifier_param_names,
                )

                grad_sim = F.cosine_similarity(grad_real, grad_syn, dim=0)
                loss_label = (1.0 - grad_sim) / num_labels
                loss_label.backward()
                step_loss += float(loss_label.detach().cpu())

            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=step_loss)


def _classifier_param_names(learner: Any) -> list[str]:
    """Return names of parameters belonging to the classifier head only."""
    out: list[str] = []
    head_names = {"classifier", "score", "sequence_summary", "logits_proj"}
    for name, _ in learner.named_parameters():
        top = name.split(".")[0]
        if top in head_names:
            out.append(name)
    return out


def _compute_grad(
    learner: Any,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    inputs: dict[str, torch.Tensor],
    classifier_param_names: list[str],
    loss_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute the (flattened) classifier-head gradient via `torch.func.grad`.

    With `loss_weights` provided, we use `loss.dot(loss_weights)` so backprop
    through the gradient flows into the generator's parameters.
    """

    def loss_fn(p, b):
        outputs = torch.func.functional_call(learner, (p, b), kwargs=inputs)
        losses = _per_sample_ce(outputs.logits, inputs["labels"])
        if loss_weights is None:
            return losses.mean()
        return losses.dot(loss_weights)

    grads = torch.func.grad(loss_fn)(params, buffers)
    flat = [grads[name].reshape(-1) for name in classifier_param_names]
    return torch.cat(flat, dim=0)


def _per_sample_ce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, labels, reduction="none")


def _tokenize_for_generator(generator, texts, label, max_length, device):
    body = list(texts)
    # Re-attach the BOS so generator.compute_loss has the same shape it saw at training.
    bos = generator.bos_tokens[label]
    annotated = [f"{bos} {t}" for t in body]
    enc = generator.tokenizer(
        annotated, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    ).to(device)
    enc["labels"] = enc["input_ids"]
    return enc


def _tokenize_for_learner(learner_tokenizer, texts, label, num_labels, max_length, device):
    enc = learner_tokenizer(
        texts, truncation=True, max_length=max_length,
        padding="max_length", return_tensors="pt",
    ).to(device)
    enc["labels"] = torch.full((len(texts),), int(label), dtype=torch.long, device=device)
    return enc


def _sample_indices(pool: list[int], n: int, rng: torch.Generator) -> list[int]:
    if n >= len(pool):
        return list(pool)
    pick = torch.randperm(len(pool), generator=rng)[:n].tolist()
    return [pool[i] for i in pick]


def _gather_text(dataset: Any, indices: list[int], sentence_keys: tuple[str, ...]) -> list[str]:
    out = []
    for idx in indices:
        example = dataset[int(idx)]
        out.append(" ".join(str(example[k]) for k in sentence_keys))
    return out


def _pop_synthetic(pool: list[str], n: int) -> list[str]:
    if len(pool) >= n:
        out = pool[:n]
        del pool[:n]
        return out
    return list(pool)  # ran out; the next interval will regenerate


def _indices_by_label(dataset: Any, label_column: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for idx, lab in enumerate(dataset[label_column]):
        out.setdefault(int(lab), []).append(idx)
    return out


class _noop_context:
    def __init__(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
