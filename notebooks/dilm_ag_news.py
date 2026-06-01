import os
import sys
from pathlib import Path

_HF_CACHE = Path(__file__).resolve().parent.parent / "hf_cache"
if _HF_CACHE.exists():
    os.environ.setdefault("HF_HOME", str(_HF_CACHE))
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    print(f"Offline mode: HF_HOME={_HF_CACHE}")

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

import mlflow
from transformers import set_seed

from dilm_wrapper import (
    build_learner, build_generator, build_data_module, build_evaluator,
    build_coreset_module, prepare_repset_teachers,
    build_trainer_lm, build_trainer_dc, generate_dilm_datasets,
    summarize, save_summary,
    RESULTS_ROOT,
)

# ── Config ────────────────────────────────────────────────────────────────────

TASK = "ag_news"
LEARNER_MODEL = "bert-base-uncased"
SEED = 42

DPC = 10       # 10 на класс × 4 класса = 40 синтетических примеров
N_DATASET = 2

EVAL_KW = dict(
    train_step=300,
    batch_size=128,
    lr=1e-4,
    n_eval_per_dataset=2,
    bf16=True,
)

# paper: LM=80k + DC=20k (~8ч на A100)
# checkpoint: LM=40k + DC=10k (~2.5ч на H100)
LM_STEPS = 40000
DC_STEPS = 10000
INNER_LOOP = 10
MODEL_STEP_PER_INNER = 20
GENERATE_INTERVAL = 10
GM_SYN_DPC = 64
GM_REAL_DPC = 100
REPSET_DPC = 100
N_REPSET = 10
OVER_SAMPLE = 50.0

RUN_NAME = f"dpc{DPC}_lm{LM_STEPS}_dc{DC_STEPS}_seed{SEED}"
dilm_dir = RESULTS_ROOT / TASK / "dilm" / RUN_NAME

set_seed(SEED)
mlflow.set_tracking_uri(f"file:{RESULTS_ROOT}/mlruns")
mlflow.set_experiment(f"dilm.{TASK}")

print(f"task: {TASK}, dpc: {DPC}, lm_steps: {LM_STEPS}, dc_steps: {DC_STEPS}")
print(f"output: {dilm_dir}")

# ── Data & models ─────────────────────────────────────────────────────────────

learner = build_learner(LEARNER_MODEL, TASK, gradient_checkpointing=True)
generator = build_generator(TASK, gradient_checkpointing=True)
data_module = build_data_module(TASK, learner, generator, train_batch_size=128)
evaluator = build_evaluator(TASK, **EVAL_KW)
METRIC_KEY = evaluator.metric_key

print(f"train:  {len(data_module.datasets['train']):>7}")
print(f"eval:   {len(data_module.datasets['validation']):>7}")  # DataModule зовёт eval-сплит 'validation'
print(f"labels: {data_module.num_labels}")

# ── Teacher coresets ──────────────────────────────────────────────────────────

kc_module = build_coreset_module(TASK, "k_centers", data_module)
repset_teachers = prepare_repset_teachers(kc_module, data_module, REPSET_DPC, N_REPSET)
print(f"{len(repset_teachers)} teacher-coresets, в каждом {len(repset_teachers[0])} примеров")

# ── LM pretrain ───────────────────────────────────────────────────────────────

lm_dir = dilm_dir / "lm"
trainer_lm = build_trainer_lm(
    lm_dir,
    total_train_step=LM_STEPS,
    val_interval=LM_STEPS,
    val_skip_step=LM_STEPS + 1,
)

with mlflow.start_run(run_name=f"dilm_lm.{RUN_NAME}"):
    mlflow.log_params({"stage": "lm", "task": TASK, "steps": LM_STEPS, "seed": SEED})
    trainer_lm.fit(
        generator=generator,
        learner=learner,
        data_module=data_module,
        evaluator=evaluator,
        repset_teachers=None,
        coreset_module=kc_module,
    )

print("LM pretrain done")

# ── DC fine-tune ──────────────────────────────────────────────────────────────

dc_dir = dilm_dir / "dc"
trainer_dc = build_trainer_dc(
    dc_dir,
    total_train_step=DC_STEPS,
    inner_loop=INNER_LOOP,
    model_step_per_inner_step=MODEL_STEP_PER_INNER,
    gm_syn_dpc=GM_SYN_DPC,
    gm_real_dpc=GM_REAL_DPC,
    repset_dpc=REPSET_DPC,
    n_repset=N_REPSET,
    generate_dataset_interval=GENERATE_INTERVAL,
    val_interval=DC_STEPS,
    val_skip_step=DC_STEPS + 1,
    log_interval=max(DC_STEPS // 10, INNER_LOOP),
    dpc=DPC,
    n_dataset=N_DATASET,
    over_sample_ratio=OVER_SAMPLE,
)

with mlflow.start_run(run_name=f"dilm_dc.{RUN_NAME}"):
    mlflow.log_params({
        "stage": "dc", "task": TASK,
        "steps": DC_STEPS, "inner_loop": INNER_LOOP, "seed": SEED,
    })
    trainer_dc.fit(
        generator=generator,
        learner=learner,
        data_module=data_module,
        evaluator=evaluator,
        repset_teachers=repset_teachers,
        coreset_module=kc_module,
    )

print("DC fine-tune done")

# ── Generate & evaluate ───────────────────────────────────────────────────────

distilled = generate_dilm_datasets(
    generator=generator,
    coreset_module=kc_module,
    dpc=DPC,
    n_dataset=N_DATASET,
    over_sample_ratio=OVER_SAMPLE,
    save_dir=dilm_dir / "distilled_datasets",
)
print(f"{len(distilled)} датасетов, по {len(distilled[0])} строк")

with mlflow.start_run(run_name=f"dilm_eval.{RUN_NAME}"):
    mlflow.log_params({
        "method": "dilm", "task": TASK, "dpc": DPC,
        "lm_steps": LM_STEPS, "dc_steps": DC_STEPS, "seed": SEED,
    })
    results = evaluator.evaluate(
        dataset_list=distilled,
        learner=learner,
        data_module=data_module,
        save_result_dir=str(dilm_dir / "metrics"),
        verbose=True,
    )

summary = summarize(
    results, METRIC_KEY,
    name=RUN_NAME, method="dilm", task=TASK,
    learner=LEARNER_MODEL, dpc=DPC, n_dataset=N_DATASET,
    lm_steps=LM_STEPS, dc_steps=DC_STEPS,
)
save_summary(summary, dilm_dir)
print("Done!", summary)
