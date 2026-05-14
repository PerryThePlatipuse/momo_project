# Project Notes

## What This Project Is

Project studies dataset distillation for text classification. Main goal: compare compact text datasets produced by different methods against full-data training.

Current focus:
- coreset baselines: `random`, `k_centers`, `herding`;
- scaling laws: metric vs distilled dataset size;
- later: DiLM text-level distillation and cross-architecture transfer.

Primary benchmark target: `AG News`. `SST-2` is useful as a fast debug dataset.

## Final Result

Expected final output:
- reproducible experiment pipeline;
- distilled datasets for several `DPC` values;
- tables with Accuracy / F1, performance gap, compression ratio, and compute time;
- scaling-law plots;
- transfer matrix across learner architectures;
- final report with interpretation.

## Code Structure

- `src/dataset_attrs.py` - dataset metadata: load args, label names, metrics.
- `src/data.py` - dataset loading, preprocessing, tokenization, dataloaders.
- `src/learner.py` - classifier model wrappers and supported model metadata.
- `src/coreset/` - coreset selection methods.
- `src/evaluator.py` - train-on-distilled-data evaluation loop.
- `notebooks/dilm_wrapper.py` - notebook-friendly builders/helpers.
- `notebooks/baselines.ipynb` - baseline experiments.
- `notebooks/scaling_laws.ipynb` - herding scaling-law experiments.
- `results/` - generated metrics, summaries, plots, and MLflow runs.

## Coding Rules

Write clean, readable code. Prefer small functions with clear names. Do not add abstraction unless it removes real duplication or makes future experiments easier.

Keep experiment code easy to extend:
- new dataset should mostly require `DATASET_ATTRS` entry;
- new learner should mostly require `MODEL_ATTRS` entry;
- new method should plug into common evaluation flow;
- summaries should use stable fields so plots can consume all runs.

Avoid overcomplication. Simple, explicit, reproducible code is better than clever code.

Do not rewrite unrelated files. Preserve existing experiment outputs unless user asks to clean them.
