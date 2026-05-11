# tests/

Lightweight tests for reusable project logic.

Tests should avoid downloading large models or datasets. Prefer deterministic toy datasets for sampling and metrics checks.

Optional pipeline smoke test:

```bash
RUN_TRAINING_SMOKE=1 pytest tests/test_training_pipeline_smoke.py -q
```

This downloads/uses AG News and a tiny Hugging Face BERT model, then trains on 100-example baseline datasets to verify the end-to-end pipeline.

