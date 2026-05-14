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
- `DiLM-main/`: vendored reference repo for **DiLM** (Maekawa 2024). Used as a documentation reference only — code is ported into `src/text_distillation/dilm/` so the runtime pipeline does not depend on this folder existing. `src/text_distillation/vanilla_lm.py` also mirrors `DiLM-main/src/generator.py` + `trainer_lm.py` for the Vanilla LM baseline.

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

## How to add a new baseline

The selection methods, models, and the experiment runner are wired together
through registries so adding a new baseline is a single-file change:

1. Implement your selection function in `src/text_distillation/distillation.py`
   (or a new module that gets imported there). Decorate it:

   ```python
   from text_distillation.distillation import register_selection

   @register_selection("herding")
   def select_herding(dataset, *, k_per_class, seed=42, label_column="label", ...):
       ...
       return dataset.select(indices)
   ```

   Contract: first positional argument is `dataset`, `seed` is keyword,
   the function returns a `datasets.Dataset` subset.

2. Either add a new section to `notebooks/baselines.ipynb` following the
   same five-step pattern (data → selection → model → train → save), or
   copy `notebooks/templates/baseline_template.ipynb` for a stand-alone
   notebook. Replace the `select_*` call with your own function.

   Parameters in CAPS_LOCK are the ones that define your experiment
   (`EXPERIMENT_PREFIX`, `K_PER_CLASS`, `DATASET_NAMES`, `MODEL_NAMES`,
   `SEED`). Training hyperparameters default to project-wide T4 settings in
   `train_text_classifier`; override them inline when needed:

   ```python
   _, metrics = train_text_classifier(
       model=model,
       tokenizer=tokenizer,
       train_dataset=train_dataset,
       eval_dataset=data.eval_dataset,
       output_dir=run_dir,
       text_columns=data.dataset_info.text_columns,
       metric_name=data.dataset_info.metric_name,
       seed=SEED,
       train_batch_size=32,   # only when needed
   )
   ```

3. Add a unit test in `tests/test_selection_registry.py` (or a method-specific
   file) confirming registration and deterministic sampling on a toy dataset.

To add a new model, register a profile in
`src/text_distillation/model/registry.py` — at minimum pick the correct
`embedding_pooling` (`"first_token"` for BERT/RoBERTa/ALBERT/DeBERTa,
`"last_token"` for XLNet, or `"mean"`). Existing notebooks pick the new model
up by adding the HF id to `MODEL_NAMES`.

To analyze results, use `text_distillation.analysis.collect_runs()` to load
all `artifacts/runs/*/{config,metrics}.json` into a pandas DataFrame.

