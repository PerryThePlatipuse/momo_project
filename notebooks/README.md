# notebooks/

Main research interface for the project.

The single `baselines.ipynb` notebook contains every baseline:

1. **01 Full-Data Baseline** — upper bound, train on the entire pool.
2. **02 Random Coreset** — uniform random sample, no class balancing.
3. **03 Stratified Random Coreset** — `K_PER_CLASS` per class.
4. **04 K-Center over TF-IDF** — lexical-space coverage.
5. **05 K-Center over Encoder Embeddings** — semantic-space coverage. Pooling is auto-selected per model (XLNet → last token).
6. **06 Herding** — Welling 2009 over encoder embeddings.

Each section reuses the same pipeline:

1. **Data** — `load_baseline_data(dataset_name, seed=SEED)`
2. **Selection** — `select_*(data.train_pool, k_per_class=..., seed=SEED)` (this is what changes between sections)
3. **Model** — `load_tokenizer(model_name)` + `load_sequence_classifier(model_name, num_labels=..., label_names=...)`
4. **Training** — `train_text_classifier(model=..., tokenizer=..., train_dataset=..., eval_dataset=..., ...)`
5. **Save** — `save_baseline_run(run_dir, data=..., method_name=..., model_name=..., ...)`

Steps 3–5 are wrapped in a small `run_all_models(...)` helper defined at the top of the notebook — visible inline, not hidden in `src/`. Selection runs once per dataset; training runs once per (dataset, model) pair.

Other notebooks:

- `00_check_setup.ipynb` — environment and import smoke check.
- `templates/baseline_template.ipynb` — starting point for a stand-alone baseline notebook; see [AGENTS.md](../AGENTS.md) section "How to add a new baseline".
