"""
Догружает недостающие для full-match ассеты в hf_cache/:
4 модели (bert-large, roberta-large, deberta-v3-base, xlnet-base) + mnli, qqp.

    python -u download_extra.py
"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).parent
HF_CACHE = PROJECT_ROOT / "hf_cache"
HF_CACHE.mkdir(exist_ok=True)
DEST_HUB = HF_CACHE / "hub"
DEST_HUB.mkdir(exist_ok=True)

from huggingface_hub import snapshot_download

MODELS = [
    "bert-large-uncased",
    "roberta-large",
    "microsoft/deberta-v3-base",
    "xlnet-base-cased",
]
DATASETS = [("glue", "mnli"), ("glue", "qqp")]

import time


def with_retry(fn, what, tries=20):
    for k in range(1, tries + 1):
        try:
            return fn()
        except Exception as exc:
            print(f"  [{what}] попытка {k} оборвалась: {type(exc).__name__}; повтор через 5с")
            time.sleep(5)
    raise RuntimeError(f"не удалось: {what}")


print("── Models ──")
for m in MODELS:
    print(f"\n[model] {m}")
    with_retry(lambda: snapshot_download(
        repo_id=m, repo_type="model",
        cache_dir=str(DEST_HUB.parent),
        ignore_patterns=[
            "*.msgpack", "flax_model*", "tf_model*", "rust_model*",
            "*.h5", "*.tflite", "*.ot", "onnx/*", "coreml/*",
        ],
    ), what=m)
    print("  done")

print("\n── Datasets ──")
os.environ["HF_HOME"] = str(HF_CACHE)
from datasets import load_dataset
for name, cfg in DATASETS:
    print(f"\n[dataset] {name}/{cfg}")
    ds = with_retry(lambda: load_dataset(name, cfg), what=f"{name}/{cfg}")
    print(f"  done: splits={list(ds.keys())}")

total = sum(f.stat().st_size for f in HF_CACHE.rglob("*") if f.is_file())
print(f"\nhf_cache total: {total/1e9:.2f} GB")
print("Done.")
