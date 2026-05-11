# notebooks/

Main research interface for the project.

Each notebook should define one experiment clearly. Put all experiment parameters near the top as `CAPS_LOCK` variables, then call reusable functions from `src/text_distillation`.

Important notebooks:

- `00_check_setup.ipynb`: environment and import smoke check.
- `01_full_data_baseline.ipynb`: train on the full AG News train split.
- `02_random_coreset_baseline.ipynb`: sample `K_TOTAL` examples without class balancing.
- `03_stratified_random_baseline.ipynb`: sample `K_PER_CLASS` examples per class.
- `04_kcenter_tfidf_baseline.ipynb`: select examples with k-center over TF-IDF features.
- `05_kcenter_embedding_baseline.ipynb`: select examples with k-center over BERT `[CLS]` embeddings.

Notebook outputs are useful during exploration but should usually be cleared before committing.
