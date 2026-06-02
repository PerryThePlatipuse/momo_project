# Scaling Laws Aggregation

This folder aggregates every scaling-law-like result found in the repo.

## Files

- `aggregate_scaling_laws.py` - dependency-free script that rebuilds all files below.
- `scaling_laws_aggregated.csv` - normalized long CSV with all available points.
- `scaling_laws_coverage.csv` - method-level coverage against the full herding grid.
- `scaling_laws_missing_grid.csv` - exact missing `(method, task, learner, dpc)` points for target methods.
- `scaling_laws_all_subplots.svg` - one SVG figure with all available primary-metric curves.

## Current Status

- `herding`: complete, 224 / 224 target points.
- `random`: partial, 32 / 224 target points.
- `k_centers`: partial, 32 / 224 target points.
- `dilm`: missing, 0 / 224 target points.
- `embedding_distillation`: extra small AG News / BERT run, included separately.

Target grid is inferred from `scaling_laws_herding_report/raw_results.csv`:
4 tasks x 7 learners x 8 DPC values.
