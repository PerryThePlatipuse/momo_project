# Project Notes

## Project

This repository studies text dataset distillation for classification tasks.

Goal: build small distilled datasets, train learner models on them, and compare quality against full-data training.

Current practical scope:
- coreset baselines: `random`, `k_centers`, `herding`;
- scaling-law runs over dataset size (`DPC`);
- `AG News` as main benchmark;
- `SST-2` as fast debug benchmark;
- later: DiLM and cross-architecture transfer.

## Expected Outcome

Final deliverables:
- reproducible experiment pipeline;
- distilled datasets for multiple budgets;
- result tables with metrics and compute costs;
- scaling-law plots;
- transferability matrix across model architectures;
- final written report.

## Repository Map

- `src/dataset_attrs.py` - dataset configs and metric metadata.
- `src/data.py` - loading, preprocessing, tokenization.
- `src/learner.py` - learner model wrapper and model metadata.
- `src/coreset/` - coreset algorithms.
- `src/evaluator.py` - evaluation loop.
- `notebooks/dilm_wrapper.py` - shared notebook helpers.
- `notebooks/baselines.ipynb` - baseline experiments.
- `notebooks/scaling_laws.ipynb` - herding scaling-law runs.
- `results/` - generated outputs.

## Near-Term Work

1. Run and validate herding scaling laws.
2. Ensure `AG News` works end to end.
3. Add full-data baseline.
4. Aggregate summaries into CSV tables.
5. Plot metric vs `K = DPC * num_labels`.
6. Keep code ready for adding new methods, datasets, and models.

## Engineering Style

Keep code beautiful, simple, and readable. Avoid clever abstractions and unnecessary complexity.

Prefer:
- explicit configs;
- small helper functions;
- stable result schemas;
- reusable experiment loops;
- clear names over comments;
- minimal changes with narrow scope.

Design for easy experiment extension:
- adding dataset = add metadata, not rewrite pipeline;
- adding model = add metadata, not duplicate evaluator;
- adding method = plug into common generation/evaluation/summarization path.

Do not touch unrelated files or delete outputs unless explicitly asked.
