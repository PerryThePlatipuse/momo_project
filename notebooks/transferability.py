"""Cross-architecture transferability.

Source encoder builds the coreset; target encoder is trained on it.
Sharded by env vars TRANSFER_TASKS / TRANSFER_METHODS so we can run
multiple GPUs in parallel without collisions.

Usage (single GPU):
  CUDA_VISIBLE_DEVICES=0 python notebooks/transferability.py

Multi-GPU split (run each line in its own tmux pane / shell):
  CUDA_VISIBLE_DEVICES=0 TRANSFER_TASKS=ag_news TRANSFER_METHODS=herding,random,k_centers python ...
  CUDA_VISIBLE_DEVICES=1 TRANSFER_TASKS=sst2    TRANSFER_METHODS=herding,random,k_centers python ...
  ...
"""

import gc
import os
import sys
import time
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "32")

_HF_CACHE = Path(__file__).resolve().parent.parent / "hf_cache"
if _HF_CACHE.exists():
    os.environ.setdefault("HF_HOME", str(_HF_CACHE))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import mlflow
from transformers import set_seed

cwd = Path.cwd().resolve()
if (cwd / "notebooks" / "dilm_wrapper.py").exists():
    ROOT = cwd
    NOTEBOOK_DIR = cwd / "notebooks"
elif (cwd / "dilm_wrapper.py").exists():
    NOTEBOOK_DIR = cwd
    ROOT = cwd.parent
else:
    raise RuntimeError(f"Cannot locate project root from {cwd}")

sys.path.insert(0, str(NOTEBOOK_DIR))
sys.path.insert(0, str(ROOT / "src"))

from coreset import CoresetConfig, CoresetModule
from dilm_wrapper import (
    RESULTS_ROOT,
    DATA_ROOT,
    build_data_module,
    build_evaluator,
    build_generator,
    build_learner,
    save_summary,
    summarize,
)

# ── Config ────────────────────────────────────────────────────────────────────

SEED = 42
DPC_GRID = [10, 100]

ALL_TASKS = ["ag_news", "sst2", "mnli", "qqp"]
ALL_METHODS = ["herding", "random", "k_centers"]

TASKS = os.environ.get("TRANSFER_TASKS", ",".join(ALL_TASKS)).split(",")
METHODS = os.environ.get("TRANSFER_METHODS", ",".join(ALL_METHODS)).split(",")

# (method, task) pairs we have verified bert-base coresets for.
# Other pairs are skipped to keep numbers honest.
ALLOWED = {
    ("herding", "ag_news"), ("herding", "sst2"),
    ("herding", "mnli"), ("herding", "qqp"),
    ("k_centers", "ag_news"), ("k_centers", "sst2"),
    ("k_centers", "mnli"), ("k_centers", "qqp"),
    ("random", "ag_news"), ("random", "sst2"),
    ("random", "mnli"), ("random", "qqp"),
}

SOURCE_MODELS = ["bert-base-uncased"]

TARGET_MODELS = [
    "bert-base-uncased",
    "roberta-base",
    "albert-base-v2",
    "bert-large-uncased",
    "roberta-large",
    # microsoft/deberta-v3-base — skipped: tokenizer requires torch>=2.6 (CVE-2025-32434)
    # xlnet-base-cased — skipped: same torch.load issue
]

EVAL_KW = dict(
    train_step=200,
    batch_size=64,
    lr=1e-4,
    n_eval_per_dataset=2,
    bf16=True,
)

N_DATASET = 1
SKIP_EXISTING = True

TRANSFER_ROOT = RESULTS_ROOT / "transferability"
TRANSFER_ROOT.mkdir(parents=True, exist_ok=True)


def safe(s: str) -> str:
    return s.replace("/", "__")


def run_dir(method: str, task: str, source: str, target: str, dpc: int) -> Path:
    return TRANSFER_ROOT / method / task / safe(source) / safe(target) / f"dpc{dpc}_seed{SEED}"


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_coresets(method: str, task: str, source: str, dpc: int, data_module):
    cfg = CoresetConfig(
        coreset_type=method,
        model_name=source,
        save_dir=str(DATA_ROOT / task / "coresets" / method),
    )
    module = CoresetModule(
        cfg,
        task,
        dataset=data_module.datasets["train"],
        generator=None,
    )
    return module.generate_dataset(dpc=dpc, n=N_DATASET)


# ── Main ──────────────────────────────────────────────────────────────────────

set_seed(SEED)
mlflow.set_tracking_uri(f"file:{RESULTS_ROOT}/mlruns")
mlflow.set_experiment("transferability")

total = len(METHODS) * len(TASKS) * len(SOURCE_MODELS) * len(TARGET_MODELS) * len(DPC_GRID)
print(f"runs total: {total}")
print(f"tasks:    {TASKS}")
print(f"methods:  {METHODS}")
print(f"sources:  {SOURCE_MODELS}")
print(f"targets:  {TARGET_MODELS}")
print(f"dpc grid: {DPC_GRID}")

done = 0
t0 = time.perf_counter()

for method in METHODS:
    for task in TASKS:
        if (method, task) not in ALLOWED:
            print(f"skip pair (not in ALLOWED): method={method}, task={task}")
            continue
        for source in SOURCE_MODELS:
            print(f"\n=== method={method}, task={task}, source={source} ===")

            # Build a SOURCE data_module just to load coresets (they're cached on
            # disk in the raw text form, so any tokenizer works to read them).
            learner_src = build_learner(source, task, gradient_checkpointing=False)
            generator_src = build_generator(task, gradient_checkpointing=False)
            data_module_src = build_data_module(task, learner_src, generator_src, train_batch_size=64)
            num_labels = data_module_src.num_labels
            del learner_src, generator_src
            cleanup()

            # Load coresets once per (method, task, dpc) - same raw text for all targets
            coresets_by_dpc = {}
            for dpc in DPC_GRID:
                coresets_by_dpc[dpc] = get_coresets(method, task, source, dpc, data_module_src)
                print(f"  dpc={dpc}: coreset size = {len(coresets_by_dpc[dpc][0])}")
            del data_module_src
            cleanup()

            for dpc in DPC_GRID:
                distilled = coresets_by_dpc[dpc]
                for target in TARGET_MODELS:
                    rd = run_dir(method, task, source, target, dpc)
                    if SKIP_EXISTING and (rd / "summary.json").exists():
                        print(f"    skip existing: target={target}")
                        done += 1
                        continue

                    rd.mkdir(parents=True, exist_ok=True)
                    try:
                        # CRITICAL: rebuild data_module with TARGET tokenizer,
                        # otherwise input_ids belong to the source vocab and
                        # the target embedding lookup explodes (CUDA assert).
                        learner = build_learner(target, task, gradient_checkpointing=False)
                        generator_t = build_generator(task, gradient_checkpointing=False)
                        data_module_t = build_data_module(task, learner, generator_t, train_batch_size=64)
                        del generator_t
                        evaluator = build_evaluator(task, **EVAL_KW)
                        metric_key = evaluator.metric_key

                        t1 = time.perf_counter()
                        with mlflow.start_run(
                            run_name=f"transfer.{method}.{task}.{safe(source)}->{safe(target)}.dpc{dpc}"
                        ):
                            mlflow.log_params({
                                "task": task, "source": source, "target": target,
                                "dpc": dpc, "method": method, "seed": SEED,
                            })
                            results = evaluator.evaluate(
                                dataset_list=distilled,
                                learner=learner,
                                data_module=data_module_t,
                                save_result_dir=str(rd / "metrics"),
                                verbose=False,
                            )
                        eval_time = time.perf_counter() - t1

                        summary = summarize(
                            results, metric_key,
                            name=f"transfer.{method}.{task}.{safe(source)}->{safe(target)}.dpc{dpc}",
                            method=method, task=task,
                            source=source, target=target,
                            learner=target,
                            dpc=dpc, k=dpc * num_labels,
                            n_dataset=N_DATASET, seed=SEED,
                            eval_time_sec=eval_time,
                        )
                        save_summary(summary, rd)

                        done += 1
                        elapsed = time.perf_counter() - t0
                        eta = elapsed / done * (total - done)
                        print(f"    [{done}/{total}] target={target}: "
                              f"{metric_key}={summary[f'{metric_key}_mean']:.4f} "
                              f"({eval_time:.0f}s, ETA {eta/3600:.1f}h)")

                        del learner, evaluator, data_module_t
                    except Exception as e:
                        print(f"    FAIL target={target}: {e!r}")
                        import traceback; traceback.print_exc()
                    cleanup()

print(f"\nDone in {(time.perf_counter() - t0)/3600:.2f}h")
