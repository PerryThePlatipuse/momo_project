"""
Scaling laws for Herding across tasks x learner models x dataset size (DPC).

Resumable, shardable across GPUs. Each (task, model, dpc) run writes a
summary.json; existing runs are skipped, so the sweep is safe to restart.

Usage
-----
Prepare raw datasets once (serial, downloads + saves raw splits):
    python scaling_laws_run.py --prepare

Run one shard pinned to a GPU (launch 8 of these, one per device):
    CUDA_VISIBLE_DEVICES=0 python scaling_laws_run.py --shard 0 --num-shards 8

Collect results into a CSV + plot once everything is done:
    python scaling_laws_run.py --collect

Herding selection is deterministic, so N_DATASET=1; run-to-run variance comes
from learner retraining, captured by n_eval_per_dataset.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from datasets import load_dataset
from filelock import FileLock
from transformers import set_seed

NOTEBOOK_DIR = Path(__file__).resolve().parent
ROOT = NOTEBOOK_DIR.parent
sys.path.insert(0, str(NOTEBOOK_DIR))
sys.path.insert(0, str(ROOT / "src"))

import mlflow

# xlnet-base-cased and microsoft/deberta-v3-base ship only torch `.bin` weights.
# transformers gates torch.load behind torch>=2.6 (CVE-2025-32434); we run 2.5.1.
# These are official, trusted checkpoints, so we relax the gate for loading them.
def _allow_bin_checkpoints() -> None:
    noop = lambda *args, **kwargs: None
    import transformers.utils.import_utils as iu
    iu.check_torch_load_is_safe = noop
    import transformers.modeling_utils as mu
    if hasattr(mu, "check_torch_load_is_safe"):
        mu.check_torch_load_is_safe = noop


_allow_bin_checkpoints()

from dataset_attrs import DATASET_ATTRS
from dilm_wrapper import (
    DATA_ROOT,
    RESULTS_ROOT,
    build_coreset_module,
    build_data_module,
    build_evaluator,
    build_generator,
    build_learner,
    save_summary,
    summarize,
)

# ── config ──────────────────────────────────────────────────────────────────

METHOD = "herding"
SEED = 42

TASKS = ["sst2", "mnli", "qqp", "ag_news"]
MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "microsoft/deberta-v3-base",
    "albert-base-v2",
    "xlnet-base-cased",
]

DPC_GRID = [1, 5, 10, 20, 50, 100, 500, 1000]
N_DATASET = 1  # herding is deterministic; more datasets would be identical

# robust evaluation: 3 retrains per dataset, 200 train steps (tight error bars)
EVAL_KW = dict(
    train_step=200,
    batch_size=64,
    lr=1e-4,
    n_eval_per_dataset=3,
    bf16=True,
)

SCALING_ROOT = RESULTS_ROOT / "scaling_laws" / METHOD


def safe_name(value: str) -> str:
    return value.replace("/", "__")


def run_dir(task: str, model: str, dpc: int) -> Path:
    return SCALING_ROOT / task / safe_name(model) / f"dpc{dpc}_seed{SEED}"


def summary_path(task: str, model: str, dpc: int) -> Path:
    return run_dir(task, model, dpc) / "summary.json"


def all_pairs() -> list[tuple[str, str]]:
    return [(task, model) for task in TASKS for model in MODELS]


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── prepare: materialize raw dataset splits (serial, race-free) ───────────────

def prepare_raw_datasets() -> None:
    """Download + save raw splits so parallel workers only read from disk.

    Mirrors DataModule.get_dataset: saves BEFORE the label->labels rename, so the
    on-disk copy keeps the original columns that get_dataset expects.
    """
    for task in TASKS:
        attr = DATASET_ATTRS[task]
        datasets_path = DATA_ROOT / task / "datasets"
        if datasets_path.exists():
            print(f"[prepare] {task}: raw datasets already present")
            continue
        print(f"[prepare] {task}: downloading {attr['load_args']}")
        datasets = load_dataset(*attr["load_args"])
        if "validation" not in datasets:
            datasets["validation"] = datasets.pop(attr["test_split_key"])
        assert datasets.keys() >= {"train", "validation"}
        datasets_path.parent.mkdir(parents=True, exist_ok=True)
        datasets.save_to_disk(str(datasets_path))
        print(f"[prepare] {task}: saved to {datasets_path}")


# ── worker: run a shard of (task, model) pairs ────────────────────────────────

def coreset_lock(task: str, dpc: int) -> FileLock:
    lock_dir = DATA_ROOT / task / "coresets" / METHOD / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return FileLock(str(lock_dir / f"dpc{dpc}.lock"))


def run_task_model(task: str, model: str, experiment: str) -> None:
    print(f"\n{'='*64}\n{task} | {model}\n{'='*64}", flush=True)

    learner = build_learner(model, task, gradient_checkpointing=False)
    generator = build_generator(task, gradient_checkpointing=False)
    data_module = build_data_module(task, learner, generator, train_batch_size=64)
    evaluator = build_evaluator(task, **EVAL_KW)
    selector = build_coreset_module(task, METHOD, data_module)
    metric_key = evaluator.metric_key

    for dpc in DPC_GRID:
        path = summary_path(task, model, dpc)
        if path.exists():
            print(f"  dpc={dpc}: skip (exists)", flush=True)
            continue

        k = dpc * data_module.num_labels
        rd = run_dir(task, model, dpc)
        rd.mkdir(parents=True, exist_ok=True)
        print(f"  dpc={dpc} (K={k}) ...", flush=True)

        t0 = time.perf_counter()
        # coreset cache is shared across models for the same (task, dpc)
        with coreset_lock(task, dpc):
            distilled = selector.generate_dataset(dpc=dpc, n=N_DATASET)
        t1 = time.perf_counter()

        with mlflow.start_run(
            run_name=f"{METHOD}.{task}.{safe_name(model)}.dpc{dpc}.seed{SEED}",
            experiment_id=experiment,
        ):
            mlflow.log_params({
                "method": METHOD, "task": task, "learner": model,
                "dpc": dpc, "k": k, "n_dataset": N_DATASET, "seed": SEED,
                **EVAL_KW,
            })
            results = evaluator.evaluate(
                dataset_list=distilled,
                learner=learner,
                data_module=data_module,
                save_result_dir=str(rd / "metrics"),
                verbose=False,
            )
        t2 = time.perf_counter()

        summary = summarize(
            results, metric_key,
            method=METHOD, task=task, learner=model,
            dpc=dpc, k=k, n_dataset=N_DATASET, seed=SEED,
            selection_time_sec=round(t1 - t0, 1),
            eval_time_sec=round(t2 - t1, 1),
        )
        save_summary(summary, rd)
        print(
            f"  dpc={dpc}: {summary[f'{metric_key}_mean']:.4f} "
            f"± {summary[f'{metric_key}_std']:.4f}  ({round(t2 - t0, 1)}s)",
            flush=True,
        )

    del learner, generator, data_module, evaluator, selector
    cleanup()


def run_shard(shard: int, num_shards: int) -> None:
    set_seed(SEED)
    mlflow.set_tracking_uri(f"file:{RESULTS_ROOT}/mlruns")
    experiment = mlflow.set_experiment(f"scaling_laws.{METHOD}.shard{shard}").experiment_id

    pairs = all_pairs()[shard::num_shards]
    print(f"[shard {shard}/{num_shards}] {len(pairs)} pairs: {pairs}", flush=True)

    for task, model in pairs:
        try:
            run_task_model(task, model, experiment)
        except Exception as exc:  # keep going; one bad model shouldn't kill the shard
            cleanup()
            err_dir = SCALING_ROOT / task / safe_name(model)
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.json").write_text(json.dumps({
                "task": task, "learner": model,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            }, indent=2))
            print(f"FAILED {task} {model}: {type(exc).__name__}: {exc}", flush=True)

    print(f"[shard {shard}/{num_shards}] done", flush=True)


# ── collect: aggregate summaries into CSV + plot ──────────────────────────────

def collect() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    rows = [json.loads(p.read_text()) for p in SCALING_ROOT.rglob("summary.json")]
    if not rows:
        print("No summaries found.")
        return

    df = pd.DataFrame(rows)
    df["score"] = df.apply(lambda r: r[f"{r['metric']}_mean"], axis=1)
    df["score_std"] = df.apply(lambda r: r[f"{r['metric']}_std"], axis=1)
    df = df.sort_values(["task", "learner", "dpc"])

    csv_path = SCALING_ROOT / "scaling_laws_herding.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}  ({len(df)} rows)")

    sns.set_theme(style="whitegrid")
    g = sns.relplot(
        data=df.astype({"learner": str}),
        x="k", y="score", hue="learner", col="task",
        kind="line", marker="o",
        facet_kws={"sharey": False, "sharex": False},
        height=4, aspect=1.1,
    )
    g.set_axis_labels("K = DPC * num_labels", "score")
    g.set_titles("{col_name}")
    g.fig.suptitle("Scaling laws for Herding", y=1.05)

    plot_path = SCALING_ROOT / "scaling_laws_herding.png"
    g.fig.savefig(plot_path, dpi=160, bbox_inches="tight")
    print(f"Saved {plot_path}")


# ── status: print progress ────────────────────────────────────────────────────

def status() -> None:
    pairs = all_pairs()
    total = len(pairs) * len(DPC_GRID)
    done = sum(summary_path(t, m, d).exists() for t, m in pairs for d in DPC_GRID)
    print(f"progress: {done}/{total} runs")
    for task in TASKS:
        for model in MODELS:
            n = sum(summary_path(task, model, d).exists() for d in DPC_GRID)
            mark = "ok " if n == len(DPC_GRID) else "   "
            print(f"  [{mark}] {task:8s} {model:28s} {n}/{len(DPC_GRID)}")


# ── cli ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="download raw datasets")
    parser.add_argument("--shard", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=8)
    parser.add_argument("--collect", action="store_true", help="aggregate CSV + plot")
    parser.add_argument("--status", action="store_true", help="print progress")
    args = parser.parse_args()

    if args.prepare:
        prepare_raw_datasets()
    elif args.collect:
        collect()
    elif args.status:
        status()
    elif args.shard is not None:
        run_shard(args.shard, args.num_shards)
    else:
        parser.error("nothing to do: pass --prepare, --shard, --collect, or --status")


if __name__ == "__main__":
    main()
