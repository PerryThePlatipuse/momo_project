# data/

Code for dataset loading, splits, dataloaders, tokenization, transforms, and tiny subsets.

This folder contains source code and should be committed.

Local dataset files belong in the repository-level `data/` folder, not here.

Supported dataset names:

- `ag_news`
- `sst2`
- `qqp`
- `mnli-m`

Use `get_dataset_info(...)` to retrieve text columns, split names, label column, and paper-style metric name.
