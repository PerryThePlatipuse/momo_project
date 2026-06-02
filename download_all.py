"""
Запусти этот скрипт локально (с интернетом) перед переносом на сервер.
Скачивает все модели и датасеты в hf_cache/ внутри проекта.

    python -u download_all.py
"""
import sys
import shutil
from pathlib import Path

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

PROJECT_ROOT = Path(__file__).parent
HF_CACHE = PROJECT_ROOT / "hf_cache"
HF_CACHE.mkdir(exist_ok=True)
DEST_HUB = HF_CACHE / "hub"
DEST_HUB.mkdir(exist_ok=True)

SYS_HUB = Path.home() / ".cache" / "huggingface" / "hub"

print(f"Project hf_cache: {HF_CACHE}")
print(f"System hub cache: {SYS_HUB}")

from huggingface_hub import snapshot_download

MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "gpt2",
]

DATASETS = [
    ("ag_news", None),
    ("glue", "sst2"),
]

# ── Step 1: copy from system cache if available ───────────────────────────────

def safe_hub_name(repo_id: str, repo_type: str = "model") -> str:
    prefix = "datasets" if repo_type == "dataset" else "models"
    return f"{prefix}--{repo_id.replace('/', '--')}"


def try_copy_from_sys_cache(hub_name: str) -> bool:
    src = SYS_HUB / hub_name
    dst = DEST_HUB / hub_name
    if dst.exists():
        print(f"  already in project cache: {hub_name}")
        return True
    if src.exists():
        sz = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
        print(f"  copying from system cache: {hub_name} ({sz / 1e6:.0f} MB)")
        shutil.copytree(src, dst)
        return True
    return False


print("\n── Models ───────────────────────────────────────────────────────────────")
for model_id in MODELS:
    hub_name = safe_hub_name(model_id, "model")
    print(f"\n[model] {model_id}")
    if try_copy_from_sys_cache(hub_name):
        continue
    print(f"  downloading from HuggingFace Hub ...")
    snapshot_download(
        repo_id=model_id,
        repo_type="model",
        cache_dir=str(DEST_HUB.parent),
        # keep safetensors; skip redundant pytorch_model.bin & non-PyTorch formats
        ignore_patterns=[
            "pytorch_model*.bin",
            "*.msgpack", "flax_model*", "tf_model*", "rust_model*",
            "*.ot",
        ],
    )
    print(f"  done")

print("\n── Datasets ─────────────────────────────────────────────────────────────")

# Datasets use load_dataset which handles configs (e.g. "glue", "sst2").
# We set HF_HOME so the downloaded files land in the project cache.
import os
os.environ["HF_HOME"] = str(HF_CACHE)
from datasets import load_dataset

for dataset_id, config in DATASETS:
    full_id = f"{dataset_id}/{config}" if config else dataset_id
    hub_name = safe_hub_name(dataset_id, "dataset")
    print(f"\n[dataset] {full_id}")
    # Check if hub snapshot already in project cache
    if (DEST_HUB / hub_name).exists():
        print(f"  already in project cache: {hub_name}")
        continue
    # Copy hub snapshot from system cache if available
    if try_copy_from_sys_cache(hub_name):
        pass
    # Always run load_dataset so processed cache is also populated
    print(f"  loading dataset (downloads if needed) ...")
    if config:
        ds = load_dataset(dataset_id, config)
    else:
        ds = load_dataset(dataset_id)
    print(f"  done: {ds}")

# ── Summary ───────────────────────────────────────────────────────────────────

total_bytes = sum(f.stat().st_size for f in HF_CACHE.rglob("*") if f.is_file())
print(f"\nProject hf_cache total: {total_bytes / 1e9:.2f} GB")
print("Done! Теперь можно паковать проект.")
