# text_distillation/

Small research library used by the experiment notebooks.

Important modules:

- `data/`: AG News loading, tokenization, dataloaders, tiny subsets.
- `model/`: model/tokenizer loading and training helpers.
- `distillation.py`: random, stratified random, TF-IDF k-center, and embedding k-center selection.
- `evaluation.py`: accuracy, macro-F1, and classifier evaluation.
- `saving.py`: result and dataset persistence.
- `utils.py`: seeds, JSON, paths, git, and device helpers.

Keep this package simple. Prefer explicit functions over registries or complex experiment runners.

