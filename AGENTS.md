# AGENTS.md

## Project Purpose

This repository is a research-first / notebook-first project for comparing text dataset distillation methods on classification tasks.

The main research questions are:

- how model quality changes when training on full vs compressed datasets;
- how quality depends on distilled dataset size;
- whether distilled datasets transfer between model architectures;
- how distillation methods compare with simple baselines.

The first working scope is AG News with three baselines:

1. Full-data baseline: train BERT/RoBERTa on the full train split.
2. Stratified Random Coreset: sample `K_PER_CLASS` examples per class.
3. K-Center over BERT `[CLS]` embeddings: embed texts, then select representative points per class.

## Core Principle

The experiment notebook defines the experiment.

- `notebooks/` is the main research interface.
- `src/text_distillation/` contains reusable logic.
- Notebooks should make it obvious what is being run.
- Method implementations must not be duplicated inside notebooks.
- Experiment parameters live at the beginning of notebooks as regular `CAPS_LOCK` variables.

Example:

```python
EXPERIMENT_NAME = "random_coreset_agnews_bert_base_k100"
DATASET_NAME = "ag_news"
K_PER_CLASS = 100
SEED = 42
```

Avoid adding config frameworks unless the team explicitly decides otherwise.

Do not add without team agreement:

- Hydra;
- OmegaConf;
- MLflow;
- complex experiment runners;
- production-style pipelines;
- heavy registry/factory systems.

If a simple function is enough, use a simple function.

## Repository Layout

- `notebooks/`: experiment notebooks and notebook README.
- `src/text_distillation/`: reusable project library.
- `src/text_distillation/data/`: dataset loading, splits, dataloaders, tokenization helpers, tiny subsets.
- `src/text_distillation/model/`: tokenizer/model loading and training helpers.
- `src/text_distillation/distillation.py`: coreset and distillation selection methods.
- `src/text_distillation/evaluation.py`: metrics and classifier evaluation.
- `src/text_distillation/saving.py`: saving configs, metrics, distilled datasets, and predictions.
- `src/text_distillation/utils.py`: seeds, paths, JSON, git helpers, device helpers.
- `data/`: local raw/processed/distilled data, usually not committed except README/placeholders.
- `artifacts/`: experiment outputs, tables, plots, runs, usually not committed except README/placeholders.
- `tests/`: small tests for reusable logic.
- `DiLM-implementation/`: upstream reference implementation from the paper. Treat as reference material, not as the main project style.

## Notebook Structure

Recommended notebook sections:

1. Short experiment goal.
2. Imports.
3. Experiment parameters via `CAPS_LOCK`.
4. Data loading.
5. Baseline or distilled dataset construction.
6. Model training.
7. Evaluation.
8. Saving results.
9. Short conclusion.

Bad:

```python
# long k-center implementation directly in notebook
```

Good:

```python
from text_distillation.distillation import select_kcenter_embeddings

distilled_dataset = select_kcenter_embeddings(
    dataset=train_dataset,
    text_column="text",
    label_column="label",
    k_per_class=K_PER_CLASS,
    model_name=EMBEDDING_MODEL_NAME,
    seed=SEED,
)
```

## Engineering Notes

- Keep APIs explicit and small.
- Prefer Hugging Face `datasets.Dataset` as the shared data object.
- Support macOS and CUDA: use `cuda` when available, then `mps`, then `cpu`.
- Do not assume CUDA-only code paths.
- Avoid expensive downloads during unit tests.
- Keep tests focused on deterministic sampling, metrics, saving, and pure functions.
- Store heavy outputs under `data/` or `artifacts/`, not in source modules.

