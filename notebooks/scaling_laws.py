import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Оффлайн-режим: если рядом есть hf_cache/ — используем его
_HF_CACHE = Path(__file__).resolve().parent.parent / "hf_cache"
if _HF_CACHE.exists():
    os.environ.setdefault("HF_HOME", str(_HF_CACHE))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    print(f"Offline mode: HF_HOME={_HF_CACHE}")

import matplotlib
matplotlib.use("Agg")
import mlflow
import pandas as pd
import seaborn as sns
import torch
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

from dilm_wrapper import (
    RESULTS_ROOT,
    build_coreset_module,
    build_data_module,
    build_evaluator,
    build_generator,
    build_learner,
    save_summary,
    summarize,
)

sns.set_theme(style="whitegrid")

# ── Config ────────────────────────────────────────────────────────────────────

# herding уже прогнан коллегой (scaling_laws_herding_report/) — не дублируем.
# Здесь добиваем random + k_centers в ТОЙ ЖЕ конфигурации для сопоставимости.
METHODS = ["random", "k_centers"]
SEED = 42

TASKS = ["sst2", "mnli", "qqp", "ag_news"]
LEARNER_MODELS = [
    "bert-base-uncased",
    "bert-large-uncased",
    "roberta-base",
    "roberta-large",
    "microsoft/deberta-v3-base",
    "albert-base-v2",
    "xlnet-base-cased",
]

# 2 методов × 4 задачи × 7 моделей × 8 DPC × 3 повтора ≈ 1344 обучения
# embeddings (для k_centers) кешируются на диск
DPC_GRID = [1, 5, 10, 20, 50, 100, 500, 1000]
N_DATASET = 1
SKIP_EXISTING = True

# точное совпадение с конфигом herding-отчёта коллеги
EVAL_KW = dict(
    train_step=200,
    batch_size=64,
    lr=1e-4,
    n_eval_per_dataset=3,
    bf16=True,
)

SCALING_ROOT = RESULTS_ROOT / "scaling_laws"
SCALING_ROOT.mkdir(parents=True, exist_ok=True)

set_seed(SEED)
mlflow.set_tracking_uri(f"file:{RESULTS_ROOT}/mlruns")
mlflow.set_experiment("scaling_laws.all_methods")

print("methods:", METHODS)
print("tasks:", TASKS)
print("models:", LEARNER_MODELS)
print("dpc:", DPC_GRID)
print("total runs:", len(METHODS) * len(TASKS) * len(LEARNER_MODELS) * len(DPC_GRID))

# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(value: str) -> str:
    return value.replace("/", "__")


def run_dir(method: str, task: str, learner_model: str, dpc: int) -> Path:
    return SCALING_ROOT / method / task / safe_name(learner_model) / f"dpc{dpc}_seed{SEED}"


def summary_path(method: str, task: str, learner_model: str, dpc: int) -> Path:
    return run_dir(method, task, learner_model, dpc) / "summary.json"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ── Main loop ─────────────────────────────────────────────────────────────────

def run_method_task_model(method: str, task: str, learner_model: str) -> list[dict]:
    print(f"\n=== {method} | {task} | {learner_model} ===")
    rows = []

    learner = build_learner(learner_model, task, gradient_checkpointing=True)
    generator = build_generator(task, gradient_checkpointing=True)
    data_module = build_data_module(task, learner, generator, train_batch_size=128)
    evaluator = build_evaluator(task, **EVAL_KW)
    selector = build_coreset_module(task, method, data_module)
    metric_key = evaluator.metric_key

    for dpc in DPC_GRID:
        path = summary_path(method, task, learner_model, dpc)
        if SKIP_EXISTING and path.exists():
            print(f"  skip existing: dpc={dpc}")
            rows.append(load_summary(path))
            continue

        k = dpc * data_module.num_labels
        rd = run_dir(method, task, learner_model, dpc)
        rd.mkdir(parents=True, exist_ok=True)
        print(f"  run dpc={dpc}, k={k}")

        t0 = time.perf_counter()
        distilled = selector.generate_dataset(dpc=dpc, n=N_DATASET)
        selection_time_sec = time.perf_counter() - t0

        t1 = time.perf_counter()
        with mlflow.start_run(run_name=f"{method}.{task}.{safe_name(learner_model)}.dpc{dpc}.seed{SEED}"):
            mlflow.log_params({
                "method": method, "task": task, "learner": learner_model,
                "dpc": dpc, "k": k, "n_dataset": N_DATASET, "seed": SEED,
                **EVAL_KW,
            })
            results = evaluator.evaluate(
                dataset_list=distilled,
                learner=learner,
                data_module=data_module,
                save_result_dir=str(rd / "metrics"),
                verbose=True,
            )
        eval_time_sec = time.perf_counter() - t1

        summary = summarize(
            results, metric_key,
            name=f"{method}_dpc{dpc}_seed{SEED}",
            method=method, task=task, learner=learner_model,
            dpc=dpc, k=k, n_dataset=N_DATASET, seed=SEED,
            selection_time_sec=selection_time_sec,
            eval_time_sec=eval_time_sec,
        )
        save_summary(summary, rd)
        rows.append(summary)
        print(f"  {metric_key}={summary[f'{metric_key}_mean']:.4f}  eval_time={eval_time_sec:.0f}s")

    del learner, generator, data_module, evaluator, selector
    cleanup_cuda()
    return rows


all_rows = []
errors = []

for method in METHODS:
    for task in TASKS:
        for learner_model in LEARNER_MODELS:
            try:
                all_rows.extend(run_method_task_model(method, task, learner_model))
            except Exception as exc:
                cleanup_cuda()
                err = {
                    "method": method, "task": task, "learner": learner_model,
                    "error_type": type(exc).__name__, "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
                errors.append(err)
                err_dir = SCALING_ROOT / method / task / safe_name(learner_model)
                err_dir.mkdir(parents=True, exist_ok=True)
                (err_dir / "error.json").write_text(json.dumps(err, indent=2))
                print("FAILED", method, task, learner_model, type(exc).__name__, exc)

df = pd.DataFrame(all_rows)
csv_path = SCALING_ROOT / "scaling_laws_all_methods.csv"
df.to_csv(csv_path, index=False)
print("saved", csv_path)
print("errors:", len(errors))

# ── Plots ─────────────────────────────────────────────────────────────────────
# Наш прогон даёт random + k_centers. herding берём из готового отчёта коллеги
# (scaling_laws_herding_report/raw_results.csv) — получаем сравнение 3 методов.

COLS = ["method", "task", "learner", "dpc", "k", "score"]

rows = [json.loads(p.read_text()) for p in SCALING_ROOT.rglob("summary.json")]
ours = pd.DataFrame(rows)
if ours.empty:
    raise RuntimeError("No summaries found")
ours["score"] = ours.apply(lambda row: row[f"{row['metric']}_mean"], axis=1)
ours = ours[COLS]

frames = [ours]
herding_csv = ROOT / "scaling_laws_herding_report" / "raw_results.csv"
if herding_csv.exists():
    herd = pd.read_csv(herding_csv)
    frames.append(herd[COLS])
    print("merged herding from", herding_csv)
else:
    print("WARNING: herding report not found, plotting only our methods")

df = pd.concat(frames, ignore_index=True).sort_values(["method", "task", "learner", "dpc"])
df.to_csv(SCALING_ROOT / "scaling_laws_all_methods.csv", index=False)

plot_df = df.copy()
plot_df["learner_short"] = plot_df["learner"].str.split("/").str[-1]

# Plot 1: сравнение 3 методов на bert-base-uncased, по задачам
bert_df = plot_df[plot_df["learner"] == "bert-base-uncased"]
g1 = sns.relplot(
    data=bert_df, x="k", y="score", hue="method", col="task", col_wrap=2,
    kind="line", marker="o",
    facet_kws={"sharey": False, "sharex": False}, height=4, aspect=1.2,
)
g1.set(xscale="log")
g1.set_axis_labels("K (примеров всего)", "score")
g1.set_titles("{col_name}")
g1.figure.suptitle("Scaling laws: сравнение методов (bert-base-uncased)", y=1.02)
plot1_path = SCALING_ROOT / "scaling_laws_methods_bert.png"
g1.figure.savefig(plot1_path, dpi=160, bbox_inches="tight")
print("saved", plot1_path)

# Plot 2: сравнение методов, усреднённое по всем архитектурам, по задачам
agg = (
    plot_df.groupby(["method", "task", "k"], as_index=False)["score"].mean()
)
g2 = sns.relplot(
    data=agg, x="k", y="score", hue="method", col="task", col_wrap=2,
    kind="line", marker="o",
    facet_kws={"sharey": False, "sharex": False}, height=4, aspect=1.2,
)
g2.set(xscale="log")
g2.set_axis_labels("K (примеров всего)", "score (среднее по моделям)")
g2.set_titles("{col_name}")
g2.figure.suptitle("Scaling laws: методы, усреднённые по архитектурам", y=1.02)
plot2_path = SCALING_ROOT / "scaling_laws_methods_avg.png"
g2.figure.savefig(plot2_path, dpi=160, bbox_inches="tight")
print("saved", plot2_path)

# Plot 3: сравнение архитектур для k_centers (наш метод), по задачам
kc_df = plot_df[plot_df["method"] == "k_centers"]
g3 = sns.relplot(
    data=kc_df, x="k", y="score", hue="learner_short", col="task", col_wrap=2,
    kind="line", marker="o",
    facet_kws={"sharey": False, "sharex": False}, height=4, aspect=1.2,
)
g3.set(xscale="log")
g3.set_axis_labels("K (примеров всего)", "score")
g3.set_titles("{col_name}")
g3.figure.suptitle("Scaling laws: сравнение архитектур (k_centers)", y=1.02)
plot3_path = SCALING_ROOT / "scaling_laws_models_kcenters.png"
g3.figure.savefig(plot3_path, dpi=160, bbox_inches="tight")
print("saved", plot3_path)
