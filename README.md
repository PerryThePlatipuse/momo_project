# Text Dataset Distillation Baselines

Research-first repository for comparing text dataset distillation methods on classification tasks.

The first target dataset is AG News. The first working baselines are:

- full-data fine-tuning baseline;
- random coreset;
- stratified random coreset;
- k-center coreset over TF-IDF features;
- k-center coreset over BERT `[CLS]` embeddings;
- herding over encoder embeddings.

The intended workflow is notebook-first: open a notebook, set experiment variables at the top, call reusable functions from `src/text_distillation`, save results under `artifacts/`.

## Supported Datasets

Reusable loaders currently support:

| Project name | Hugging Face source | Text columns | Eval split | Paper metric |
|---|---|---|---|---|
| `ag_news` | `ag_news` | `text` | `test` | accuracy |
| `sst2` | `glue`, `sst2` | `sentence` | `validation` | accuracy |
| `qqp` | `glue`, `qqp` | `question1`, `question2` | `validation` | average of accuracy and binary F1 |
| `mnli-m` | `glue`, `mnli` | `premise`, `hypothesis` | `validation_matched` | accuracy |

Example:

```python
from text_distillation.data import get_dataset_info, get_train_eval_splits, load_text_classification_dataset

DATASET_NAME = "qqp"
info = get_dataset_info(DATASET_NAME)
dataset = load_text_classification_dataset(DATASET_NAME)
train_dataset, eval_dataset = get_train_eval_splits(dataset, DATASET_NAME)
```

## Setup

The project is developed against the `marketing` conda environment:

```bash
conda activate marketing
pip install -r requirements.txt
pip install -e .
```

The code supports CUDA and CPU. Large experiments are expected to be much faster on CUDA.

## T4 Defaults

Baseline notebooks are configured for a NVIDIA T4 16GB starting point:

- `TRAIN_BATCH_SIZE = 64`
- `EVAL_BATCH_SIZE = 128`
- `MIXED_PRECISION = "auto"`, which enables fp16 on CUDA
- `DATALOADER_NUM_WORKERS = 2`
- `GRADIENT_ACCUMULATION_STEPS = 1`

If a full run hits CUDA OOM, lower `TRAIN_BATCH_SIZE` to `32` first. If GPU utilization is still low and memory is available, try `TRAIN_BATCH_SIZE = 96` or `128`.

## Quick Check

```bash
pytest -q
```

Optional end-to-end training smoke test:

```bash
RUN_TRAINING_SMOKE=1 pytest tests/test_training_pipeline_smoke.py -q
```

## Project Layout

- `notebooks/`: experiment notebooks.
- `src/text_distillation/`: reusable Python code.
- `data/`: local datasets and distilled subsets.
- `artifacts/`: experiment outputs.
- `tests/`: lightweight tests for reusable logic.
- `DiLM-implementation/`: reference implementation from the DiLM paper, kept separate from this project's simpler notebook-first code.
