"""DiLM (Maekawa et al., NAACL Findings 2024) full pipeline.

Ported from `DiLM-main/src/`. Three-phase training (Vanilla LM → DC → sample),
self-contained — no dependency on the DiLM-main folder at runtime.

Public API:

- `distill_dilm(dataset, *, dataset_name, k_per_class, ...) -> Dataset`
  Train a class-conditional generator from scratch on real text, fine-tune
  it with gradient matching against a downstream classifier, sample
  `k_per_class * num_labels` synthetic examples, return a `datasets.Dataset`.

Defaults are tuned for quick smoke tests (5k LM + 2k DC steps, a few minutes
on GPU). For paper-faithful runs bump `lm_train_steps` to 80k and
`dc_train_steps` to 20k.
"""

from __future__ import annotations

from typing import Any

from datasets import Dataset

from text_distillation.data.transforms import TextColumns, normalize_text_columns
from text_distillation.distillation import register_selection
from text_distillation.timing import TimingTracker
from text_distillation.utils import get_device, set_seed

from .dc_trainer import train_generator_dc
from .generator import GeneratorConfig, GeneratorModel
from .lm_trainer import train_generator_lm
from .official import (
    OfficialDiLMRun,
    distill_dilm_official,
    load_official_dilm_dataset,
    official_dilm_paths,
    run_official_dilm_reproduction,
)

__all__ = [
    "GeneratorConfig",
    "GeneratorModel",
    "OfficialDiLMRun",
    "distill_dilm",
    "distill_dilm_official",
    "load_official_dilm_dataset",
    "official_dilm_paths",
    "run_official_dilm_reproduction",
    "train_generator_dc",
    "train_generator_lm",
]


@register_selection("dilm")
def distill_dilm(
    dataset: Any,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    label_column: str = "label",
    k_per_class: int = 20,
    seed: int = 42,
    # Generator.
    generator_model_name: str = "gpt2",
    generate_max_length: int = 64,
    # LM phase.
    lm_train_steps: int = 5_000,
    lm_batch_size: int = 16,
    lm_learning_rate: float = 1e-5,
    lm_max_finetune_samples: int | None = 8_000,
    # DC phase.
    dc_train_steps: int = 2_000,
    dc_learning_rate: float = 3e-7,
    gm_real_dpc: int = 64,
    gm_syn_dpc: int = 16,
    generate_interval: int = 20,
    learner_model_name: str = "bert-base-uncased",
    tracker: TimingTracker | None = None,
    **_unused,
) -> Any:
    """Full DiLM pipeline on `dataset` → synthetic Dataset.

    `dataset` must be a `datasets.Dataset` with `text_column(s)` and `label_column`.
    Returns a `datasets.Dataset` with the same schema.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_seed(seed)
    device = get_device()

    columns = normalize_text_columns(text_column=text_column, text_columns=text_columns)
    labels_seen = sorted(set(int(x) for x in dataset[label_column]))
    num_labels = len(labels_seen)
    if labels_seen != list(range(num_labels)):
        raise ValueError(
            f"DiLM expects labels to be 0..num_labels-1, got {labels_seen}"
        )

    # ---- Build the class-conditional generator ----
    gen_config = GeneratorConfig(
        model_name=generator_model_name,
        generate_max_length=generate_max_length,
    )
    generator = GeneratorModel(gen_config, num_labels=num_labels, sentence_keys=tuple(columns))
    generator.to(device)

    # ---- Phase 1: standard causal-LM fine-tune ----
    train_generator_lm(
        generator,
        dataset,
        label_column=label_column,
        max_steps=lm_train_steps,
        batch_size=lm_batch_size,
        learning_rate=lm_learning_rate,
        max_finetune_samples=lm_max_finetune_samples,
        seed=seed,
        tracker=tracker,
    )

    # ---- Phase 2: gradient-matching DC fine-tune ----
    learner_tokenizer = AutoTokenizer.from_pretrained(learner_model_name)
    learner = AutoModelForSequenceClassification.from_pretrained(
        learner_model_name, num_labels=num_labels,
    ).to(device)

    train_generator_dc(
        generator,
        learner=learner,
        learner_tokenizer=learner_tokenizer,
        dataset=dataset,
        label_column=label_column,
        max_steps=dc_train_steps,
        gm_real_dpc=gm_real_dpc,
        gm_syn_dpc=gm_syn_dpc,
        generate_interval=generate_interval,
        learning_rate=dc_learning_rate,
        generate_max_length=generate_max_length,
        seed=seed,
        tracker=tracker,
    )
    del learner
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Phase 3: sample k_per_class per label ----
    generator.eval()
    out: dict[str, list] = {col: [] for col in columns}
    out[label_column] = []
    _measure = tracker.measure if tracker is not None else _noop_context
    with _measure("dilm_sample_sec"):
        for label in range(num_labels):
            texts = generator.sample(label, n=k_per_class)
            for text in texts:
                decoded = generator.decode_to_example(text)
                for col in columns:
                    out[col].append(decoded.get(col, " "))
                out[label_column].append(label)

    return Dataset.from_dict(out)


class _noop_context:
    def __init__(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
