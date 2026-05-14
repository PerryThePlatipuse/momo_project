"""Thin builders + helpers around DiLM for notebook use.

No orchestration — notebook drives the pipeline step-by-step.

Extend:
    - new dataset: add entry to src/dataset_attrs.py::DATASET_ATTRS
    - new learner: add entry to src/learner.py::MODEL_ATTRS
    - new distillation method: write a function that returns list[Dataset]
      with columns {sentence(s), labels}; feed it to evaluator.evaluate()
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coreset import CoresetConfig, CoresetModule  # noqa: E402
from data import DataConfig, DataModule  # noqa: E402
from distillation import DistilledDataConfig, TrainConfig, TrainerDC, TrainerLM  # noqa: E402
from evaluator import EvaluateConfig, Evaluator  # noqa: E402
from generator import GeneratorConfig, GeneratorModel  # noqa: E402
from learner import LearnerConfig, LearnerModel  # noqa: E402

RESULTS_ROOT = ROOT / "results"
DATA_ROOT = ROOT / "data"


def build_learner(model_name: str, task: str, gradient_checkpointing: bool = True) -> LearnerModel:
    cfg = LearnerConfig(model_name=model_name, gradient_checkpointing=gradient_checkpointing)
    return LearnerModel(cfg, task_name=task)


def build_generator(
    task: str,
    model_name: str = "gpt2",
    pretrained_dir: str | None = None,
    gradient_checkpointing: bool = True,
) -> GeneratorModel:
    cfg = GeneratorConfig(
        model_name=model_name,
        pretrained_model_dir=pretrained_dir,
        generate_bf16=True,
        gradient_checkpointing=gradient_checkpointing,
    )
    return GeneratorModel(cfg, task_name=task)


def build_data_module(
    task: str,
    learner: LearnerModel,
    generator: GeneratorModel,
    train_batch_size: int = 64,
) -> DataModule:
    cfg = DataConfig(
        task_name=task,
        datasets_path=str(DATA_ROOT / task / "datasets"),
        preprocessed_datasets_path=str(
            DATA_ROOT / task / f"datasets_{generator.config.model_name}_{learner.config.model_name}"
        ),
        train_batch_size=train_batch_size,
    )
    return DataModule(cfg, generator=generator, learner=learner)


def build_evaluator(
    task: str,
    train_step: int = 200,
    batch_size: int = 64,
    lr: float = 1e-4,
    n_eval_per_dataset: int = 3,
    bf16: bool = True,
    save_result_dir: str | Path | None = None,
) -> Evaluator:
    cfg = EvaluateConfig(
        task_name=task,
        n_eval_per_dataset=n_eval_per_dataset,
        bf16=bf16,
        save_result_dir=str(save_result_dir or RESULTS_ROOT / "_tmp_eval"),
        lr=lr,
        train_step=train_step,
        batch_size=batch_size,
    )
    return Evaluator(cfg, task_name=task)


def build_coreset_module(
    task: str,
    kind: str,
    data_module: DataModule,
    generator: GeneratorModel | None = None,
) -> CoresetModule:
    """Build a coreset selector. `kind` in {random, k_centers, herding, rank_dilm}."""
    cfg = CoresetConfig(coreset_type=kind, save_dir=str(DATA_ROOT / task / "coresets" / kind))
    return CoresetModule(
        cfg,
        task,
        dataset=data_module.datasets["train"],
        generator=generator if kind == "rank_dilm" else None,
    )


def prepare_repset_teachers(
    coreset_module: CoresetModule,
    data_module: DataModule,
    repset_dpc: int,
    n_repset: int,
) -> list[Dataset]:
    """Select n_repset coresets of repset_dpc examples per class, preprocess for trainer."""
    raw = coreset_module.generate_dataset(dpc=repset_dpc, n=n_repset)
    return [data_module.preprocess_dataset(d) for d in raw]


def build_trainer_lm(
    save_dir: str | Path,
    *,
    total_train_step: int = 80000,
    lr: float = 1e-5,
    lm_batch_size: int = 64,
    val_interval: int = 5000,
    val_skip_step: int = 1,
    log_interval: int = 100,
    bf16: bool = True,
) -> TrainerLM:
    """Trainer for stage-1 causal-LM pretraining of the generator on real sentences."""
    save_dir = Path(save_dir)
    train = TrainConfig(
        train_type="lm",
        lm_batch_size=lm_batch_size,
        total_train_step=total_train_step,
        lr=lr,
        optimizer_type="adamw",
        scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        max_grad_norm=1.0,
        val_interval=val_interval,
        val_skip_step=val_skip_step,
        log_interval=log_interval,
        save_model_dir=str(save_dir / "generator"),
        save_valid_result_dir=str(save_dir / "valid_results"),
        bf16=bf16,
        classifier_grad_only=False,
        use_generated_data=True,
        normalize_temperature=1.0,
    )
    # DistilledDataConfig is needed for mid-training validation only
    dd = DistilledDataConfig(
        dpc=20,
        n_dataset=5,
        save_dataset_path=str(save_dir / "dataset"),
        over_sample_ratio=100.0,
    )
    return TrainerLM(train, dd)


def build_trainer_dc(
    save_dir: str | Path,
    *,
    total_train_step: int = 20000,
    inner_loop: int = 10,
    model_step_per_inner_step: int = 20,
    gm_syn_dpc: int = 64,
    gm_real_dpc: int = 200,
    gm_real_grad_accum_step: int = 1,
    repset_dpc: int = 200,
    n_repset: int = 10,
    generate_dataset_interval: int = 20,
    lr: float = 3e-7,
    val_interval: int = 2000,
    val_skip_step: int = 0,
    log_interval: int = 100,
    bf16: bool = True,
    classifier_grad_only: bool = True,
    n_clusters_for_real_sampler: int = 1,
    n_clusters_for_syn_sampler: int | None = None,
    lm_lambda: float = 0.0,
    dpc: int = 20,
    n_dataset: int = 5,
    over_sample_ratio: float = 100.0,
) -> TrainerDC:
    """Trainer for stage-2 gradient-matching fine-tune of the generator against BERT."""
    save_dir = Path(save_dir)
    if n_clusters_for_syn_sampler is None:
        n_clusters_for_syn_sampler = gm_syn_dpc
    train = TrainConfig(
        train_type="dc",
        gm_syn_dpc=gm_syn_dpc,
        gm_real_dpc=gm_real_dpc,
        gm_real_grad_accum_step=gm_real_grad_accum_step,
        lm_lambda=lm_lambda,
        lm_batch_size=64,
        repset_teacher=True,
        repset_dpc=repset_dpc,
        n_repset=n_repset,
        classifier_grad_only=classifier_grad_only,
        normalize_temperature=1.0,
        n_clusters_for_real_sampler=n_clusters_for_real_sampler,
        n_clusters_for_syn_sampler=n_clusters_for_syn_sampler,
        use_generated_data=True,
        total_train_step=total_train_step,
        inner_loop=inner_loop,
        model_step_per_inner_step=model_step_per_inner_step,
        generate_dataset_interval=generate_dataset_interval,
        lr=lr,
        optimizer_type="adamw",
        scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        val_interval=val_interval,
        val_skip_step=val_skip_step,
        log_interval=log_interval,
        save_model_dir=str(save_dir / "generator"),
        save_valid_result_dir=str(save_dir / "valid_results"),
        bf16=bf16,
    )
    dd = DistilledDataConfig(
        dpc=dpc,
        n_dataset=n_dataset,
        save_dataset_path=str(save_dir / "dataset"),
        over_sample_ratio=over_sample_ratio,
    )
    return TrainerDC(train, dd)


def generate_dilm_datasets(
    generator: GeneratorModel,
    coreset_module: CoresetModule,
    dpc: int,
    n_dataset: int,
    over_sample_ratio: float = 100.0,
    save_dir: str | Path | None = None,
) -> list[Dataset]:
    """Sample `dpc*over_sample_ratio` per class, then prune to `dpc` with the coreset selector."""
    gen_dpc = int(dpc * over_sample_ratio)
    raw = generator.generate_dataset(dpc=gen_dpc, n=n_dataset)
    pruned = (
        [coreset_module.get_coreset(d, dpc=dpc) for d in raw]
        if over_sample_ratio > 1.0
        else raw
    )
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i, ds in enumerate(pruned):
            ds.to_json(str(save_dir / f"dataset_{i}.json"))
    return pruned


def summarize(results: list[dict], metric_key: str, **meta) -> dict:
    """Aggregate per-eval-run results into mean/std + attach metadata."""
    scores = [r[metric_key] for r in results]
    return {
        **meta,
        "metric": metric_key,
        f"{metric_key}_mean": float(np.mean(scores)),
        f"{metric_key}_std": float(np.std(scores)),
        "scores": scores,
    }


def save_summary(summary: dict, run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def collect_summaries(task: str, results_root: Path = RESULTS_ROOT) -> pd.DataFrame:
    root = Path(results_root) / task
    if not root.exists():
        return pd.DataFrame()
    rows = [json.loads(p.read_text()) for p in root.rglob("summary.json")]
    return pd.DataFrame(rows)
