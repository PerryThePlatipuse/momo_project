from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import Dataset

from text_distillation.data.datasets import get_dataset_info
from text_distillation.distillation import register_selection


PAPER_LM_STEPS = 80_000
PAPER_DC_STEPS = 20_000
PAPER_SYN_DPC = 64
PAPER_N_REPSET = 10
PAPER_N_DATASETS = 20


@dataclass(frozen=True)
class OfficialDiLMRun:
    task_name: str
    dpc: int
    output_root: Path
    data_root: Path
    lm_save_dir: Path
    lm_generator_dir: Path
    dc_save_dir: Path
    dc_generator_dir: Path
    test_save_dir: Path
    dataset_dir: Path
    result_dir: Path


def official_dilm_paths(
    dataset_name: str,
    *,
    dpc: int = 20,
    seed: int = 42,
    output_root: str | Path = "artifacts/dilm_official",
    data_root: str | Path = "data/dilm_official",
) -> OfficialDiLMRun:
    task_name = _official_task_name(dataset_name)
    output_root = Path(output_root).resolve()
    data_root = Path(data_root).resolve()

    train_experiment = f"train.gpt2.bert-base-uncased.{task_name}"
    test_experiment = f"test.bert-base-uncased.{task_name}"
    lm_subrun = f"step_{PAPER_LM_STEPS}_seed_{seed}"
    dc_subrun = f"dpc_{dpc}_seed_{seed}_paper"
    test_subrun = f"dpc_{dpc}_seed_{seed}_paper"

    lm_save_dir = output_root / train_experiment / "dilm.lm" / lm_subrun
    dc_save_dir = output_root / train_experiment / "dilm.dc" / dc_subrun
    test_save_dir = output_root / test_experiment / "dilm.dc" / test_subrun

    return OfficialDiLMRun(
        task_name=task_name,
        dpc=dpc,
        output_root=output_root,
        data_root=data_root,
        lm_save_dir=lm_save_dir,
        lm_generator_dir=lm_save_dir / "generator",
        dc_save_dir=dc_save_dir,
        dc_generator_dir=dc_save_dir / "generator",
        test_save_dir=test_save_dir,
        dataset_dir=test_save_dir / "dataset",
        result_dir=test_save_dir / "final_results",
    )


def run_official_dilm_reproduction(
    dataset_name: str,
    *,
    dpc: int = 20,
    seed: int = 42,
    output_root: str | Path = "artifacts/dilm_official",
    data_root: str | Path = "data/dilm_official",
    n_datasets: int = PAPER_N_DATASETS,
    force: bool = False,
    run_lm: bool = True,
    run_dc: bool = True,
    run_test: bool = True,
    python_executable: str | Path | None = None,
    dilm_root: str | Path | None = None,
) -> OfficialDiLMRun:
    """Run official DiLM protocol from the vendored `DiLM-main` code.

    This does not load pregenerated synthetic data. It trains LM, trains DC,
    then uses the trained generator to create/evaluate synthetic datasets.
    Designed for CUDA machines (V100/A100); the official code calls `.cuda()`.
    """
    run = official_dilm_paths(
        dataset_name,
        dpc=dpc,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
    )
    dilm_root = _dilm_root(dilm_root)
    python_executable = str(python_executable or sys.executable)

    if run_lm and (force or not (run.lm_generator_dir / "last-ckpt").exists()):
        _run_official_command(
            [
                python_executable,
                "src/train.py",
                "--config-name=lm",
                f"data.task_name={run.task_name}",
                f"base.seed={seed}",
                f"base.save_dir_root={run.output_root}",
                f"base.data_dir_root={run.data_root}",
                f"base.sub_run_name={run.lm_save_dir.name}",
                f"train.total_train_step={PAPER_LM_STEPS}",
                "train.lm_batch_size=64",
                "train.bf16=True",
                "train.fp16=False",
                "generator.generate_bf16=True",
                "generator.generate_fp16=False",
            ],
            cwd=dilm_root,
            env=_official_env(run.output_root),
        )

    if run_dc and (force or not (run.dc_generator_dir / "last-ckpt").exists()):
        _require_path(run.lm_generator_dir / "last-ckpt", "LM checkpoint")
        gm_real_dpc = _paper_gm_real_dpc(run.task_name)
        _run_official_command(
            [
                python_executable,
                "src/train.py",
                "--config-name=dc",
                f"data.task_name={run.task_name}",
                f"base.seed={seed}",
                f"base.save_dir_root={run.output_root}",
                f"base.data_dir_root={run.data_root}",
                f"base.sub_run_name={run.dc_save_dir.name}",
                f"generator.pretrained_model_dir={run.lm_generator_dir}",
                "generator.checkpoint_name=last-ckpt",
                f"train.total_train_step={PAPER_DC_STEPS}",
                "train.inner_loop=10",
                "train.model_step_per_inner_step=20",
                f"train.gm_real_dpc={gm_real_dpc}",
                f"train.repset_dpc={gm_real_dpc}",
                "train.gm_real_grad_accum_step=1",
                f"train.gm_syn_dpc={PAPER_SYN_DPC}",
                f"train.n_clusters_for_syn_sampler={PAPER_SYN_DPC}",
                "train.repset_teacher=True",
                f"train.n_repset={PAPER_N_REPSET}",
                "train.classifier_grad_only=True",
                "train.lm_lambda=0.0",
                "train.lr=3e-7",
                "train.warmup_ratio=0.05",
                "train.bf16=True",
                "train.fp16=False",
                "generator.generate_bf16=True",
                "generator.generate_fp16=False",
                f"distilled_data.dpc={dpc}",
                "distilled_data.over_sample_ratio=100.0",
            ],
            cwd=dilm_root,
            env=_official_env(run.output_root),
        )

    if run_test and (force or not _has_generated_datasets(run.dataset_dir, n_datasets)):
        _require_path(run.dc_generator_dir / "last-ckpt", "DC checkpoint")
        _run_official_command(
            [
                python_executable,
                "src/test.py",
                "--config-name=dc",
                f"data.task_name={run.task_name}",
                f"base.seed={seed}",
                f"base.save_dir_root={run.output_root}",
                f"base.data_dir_root={run.data_root}",
                f"base.sub_run_name={run.test_save_dir.name}",
                f"generator.pretrained_model_dir={run.dc_generator_dir}",
                "generator.checkpoint_name=last-ckpt",
                f"distilled_data.dpc={dpc}",
                f"distilled_data.n_dataset={n_datasets}",
                "distilled_data.over_sample_ratio=1.0",
                f"distilled_data.save_dataset_path={run.dataset_dir}",
                "evaluate.n_eval_per_dataset=5",
                "evaluate.train_step=200",
                "evaluate.batch_size=64",
                "evaluate.bf16=True",
                "evaluate.fp16=False",
                f"evaluate.save_result_dir={run.result_dir}",
            ],
            cwd=dilm_root,
            env=_official_env(run.output_root),
        )

    return run


def load_official_dilm_dataset(
    dataset_name: str,
    *,
    dpc: int = 20,
    seed: int = 42,
    dataset_index: int = 0,
    output_root: str | Path = "artifacts/dilm_official",
    data_root: str | Path = "data/dilm_official",
) -> Dataset:
    run = official_dilm_paths(
        dataset_name,
        dpc=dpc,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
    )
    path = run.dataset_dir / f"dataset_{dataset_index}.json"
    _require_path(path, "generated DiLM dataset")
    dataset = Dataset.from_json(str(path))
    info = get_dataset_info(dataset_name)
    if "labels" in dataset.column_names and info.label_column != "labels":
        dataset = dataset.rename_column("labels", info.label_column)
    return dataset


@register_selection("dilm_official")
def distill_dilm_official(
    dataset: Any | None = None,
    *,
    dataset_name: str,
    k_per_class: int = 20,
    seed: int = 42,
    dataset_index: int = 0,
    output_root: str | Path = "artifacts/dilm_official",
    data_root: str | Path = "data/dilm_official",
    n_datasets: int = PAPER_N_DATASETS,
    force: bool = False,
    python_executable: str | Path | None = None,
    dilm_root: str | Path | None = None,
    **_unused,
) -> Dataset:
    """Paper-faithful DiLM selection using official training code.

    The positional `dataset` is accepted for registry compatibility, but the
    official pipeline loads the GLUE dataset itself to match the paper protocol.
    """
    run_official_dilm_reproduction(
        dataset_name,
        dpc=k_per_class,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
        n_datasets=n_datasets,
        force=force,
        python_executable=python_executable,
        dilm_root=dilm_root,
    )
    return load_official_dilm_dataset(
        dataset_name,
        dpc=k_per_class,
        seed=seed,
        dataset_index=dataset_index,
        output_root=output_root,
        data_root=data_root,
    )


def _official_task_name(dataset_name: str) -> str:
    name = get_dataset_info(dataset_name).name
    if name in {"sst2", "qqp", "mnli"}:
        return name
    if name == "mnli-m":
        return "mnli"
    raise ValueError("Official DiLM reproduction supports only sst2, qqp, and mnli-m.")


def _paper_gm_real_dpc(task_name: str) -> int:
    return 200 if task_name == "sst2" else 100


def _dilm_root(path: str | Path | None) -> Path:
    if path is not None:
        root = Path(path).resolve()
    else:
        root = Path(__file__).resolve().parents[3] / "DiLM-main"
    _require_path(root / "src" / "train.py", "DiLM-main train.py")
    return root


def _official_env(output_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("MLFLOW_TRACKING_URI", str((output_root / "mlruns").resolve()))
    return env


def _run_official_command(command: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    subprocess.run(command, cwd=cwd, env=env, check=True)


def _has_generated_datasets(dataset_dir: Path, n_datasets: int) -> bool:
    return all((dataset_dir / f"dataset_{index}.json").exists() for index in range(n_datasets))


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
