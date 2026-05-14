from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets
from sklearn.cluster import KMeans
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    get_scheduler,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from text_distillation.data.datasets import (
    get_dataset_info,
    get_train_eval_splits,
    load_text_classification_dataset,
)
from text_distillation.distillation import register_selection
from text_distillation.evaluation import compute_classification_metrics
from text_distillation.utils import ensure_dir, set_seed

from .generator import GeneratorConfig, GeneratorModel, SEP_TOKEN


PAPER_LM_STEPS = 80_000
PAPER_DC_STEPS = 20_000
PAPER_SYN_DPC = 64
PAPER_N_REPSET = 10
PAPER_N_DATASETS = 20
PAPER_EVAL_PER_DATASET = 5

TASK_MAX_LENGTH = {"sst2": 68, "qqp": 313, "mnli": 421}


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


@dataclass(frozen=True)
class PaperVanillaLMRun:
    task_name: str
    dpc: int
    output_root: Path
    data_root: Path
    lm_save_dir: Path
    lm_generator_dir: Path
    test_save_dir: Path
    dataset_dir: Path
    result_dir: Path


@dataclass
class PaperDiLMConfig:
    dataset_name: str
    dpc: int = 20
    seed: int = 42
    output_root: str | Path = "artifacts/dilm_official"
    data_root: str | Path = "data/dilm_official"
    generator_model_name: str = "gpt2"
    learner_model_name: str = "bert-base-uncased"
    lm_train_steps: int = PAPER_LM_STEPS
    lm_batch_size: int = 64
    lm_learning_rate: float = 1e-5
    dc_train_steps: int = PAPER_DC_STEPS
    dc_learning_rate: float = 3e-7
    gm_syn_dpc: int = PAPER_SYN_DPC
    gm_real_dpc: int | None = None
    n_repset: int = PAPER_N_REPSET
    inner_loop: int = 10
    model_step_per_inner_step: int = 20
    generate_dataset_interval: int = 20
    n_datasets: int = PAPER_N_DATASETS
    n_eval_per_dataset: int = PAPER_EVAL_PER_DATASET
    eval_train_steps: int = 200
    eval_batch_size: int = 64
    bf16: bool = True
    fp16: bool = False
    save_model: bool = True


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
    lm_save_dir = output_root / train_experiment / "dilm.lm" / f"step_80000_seed_{seed}"
    dc_save_dir = output_root / train_experiment / "dilm.dc" / f"dpc_{dpc}_seed_{seed}_paper"
    test_save_dir = output_root / test_experiment / "dilm.dc" / f"dpc_{dpc}_seed_{seed}_paper"
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


def paper_vanilla_lm_paths(
    dataset_name: str,
    *,
    dpc: int = 20,
    seed: int = 42,
    output_root: str | Path = "artifacts/vanilla_lm_paper",
    data_root: str | Path = "data/vanilla_lm_paper",
) -> PaperVanillaLMRun:
    task_name = _official_task_name(dataset_name)
    output_root = Path(output_root).resolve()
    data_root = Path(data_root).resolve()
    train_experiment = f"train.gpt2.bert-base-uncased.{task_name}"
    test_experiment = f"test.bert-base-uncased.{task_name}"
    lm_save_dir = output_root / train_experiment / "dilm.lm" / f"step_80000_seed_{seed}"
    test_save_dir = output_root / test_experiment / "dilm.lm" / f"dpc_{dpc}_seed_{seed}_paper"
    return PaperVanillaLMRun(
        task_name=task_name,
        dpc=dpc,
        output_root=output_root,
        data_root=data_root,
        lm_save_dir=lm_save_dir,
        lm_generator_dir=lm_save_dir / "generator",
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
    **kwargs: Any,
) -> OfficialDiLMRun:
    """Run paper DiLM locally, without Hydra/MLflow or pregenerated data."""
    config = PaperDiLMConfig(
        dataset_name=dataset_name,
        dpc=dpc,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
        n_datasets=n_datasets,
        **{k: v for k, v in kwargs.items() if k in PaperDiLMConfig.__dataclass_fields__},
    )
    run = official_dilm_paths(
        dataset_name,
        dpc=dpc,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
    )
    if force or not _has_generated_datasets(run.dataset_dir, n_datasets):
        _run_local_paper_dilm(config, run)
    return run


def run_paper_vanilla_lm_reproduction(
    dataset_name: str,
    *,
    dpc: int = 20,
    seed: int = 42,
    output_root: str | Path = "artifacts/vanilla_lm_paper",
    data_root: str | Path = "data/vanilla_lm_paper",
    n_datasets: int = PAPER_N_DATASETS,
    force: bool = False,
    **kwargs: Any,
) -> PaperVanillaLMRun:
    """Run paper Vanilla LM locally: 80k LM steps, then sample/evaluate."""
    config = PaperDiLMConfig(
        dataset_name=dataset_name,
        dpc=dpc,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
        n_datasets=n_datasets,
        **{k: v for k, v in kwargs.items() if k in PaperDiLMConfig.__dataclass_fields__},
    )
    run = paper_vanilla_lm_paths(
        dataset_name,
        dpc=dpc,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
    )
    if force or not _has_generated_datasets(run.dataset_dir, n_datasets):
        _run_local_paper_vanilla_lm(config, run)
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
    if not path.exists():
        raise FileNotFoundError(f"generated DiLM dataset not found: {path}")
    dataset = Dataset.from_json(str(path))
    info = get_dataset_info(dataset_name)
    if "labels" in dataset.column_names and info.label_column != "labels":
        dataset = dataset.rename_column("labels", info.label_column)
    return dataset


def load_paper_vanilla_lm_dataset(
    dataset_name: str,
    *,
    dpc: int = 20,
    seed: int = 42,
    dataset_index: int = 0,
    output_root: str | Path = "artifacts/vanilla_lm_paper",
    data_root: str | Path = "data/vanilla_lm_paper",
) -> Dataset:
    run = paper_vanilla_lm_paths(
        dataset_name,
        dpc=dpc,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
    )
    path = run.dataset_dir / f"dataset_{dataset_index}.json"
    if not path.exists():
        raise FileNotFoundError(f"generated Vanilla LM dataset not found: {path}")
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
    **kwargs: Any,
) -> Dataset:
    run_official_dilm_reproduction(
        dataset_name,
        dpc=k_per_class,
        seed=seed,
        output_root=output_root,
        data_root=data_root,
        n_datasets=n_datasets,
        force=force,
        **kwargs,
    )
    return load_official_dilm_dataset(
        dataset_name,
        dpc=k_per_class,
        seed=seed,
        dataset_index=dataset_index,
        output_root=output_root,
        data_root=data_root,
    )


def _run_local_paper_vanilla_lm(config: PaperDiLMConfig, run: PaperVanillaLMRun) -> None:
    set_seed(config.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = _cuda_device()
    amp_dtype, use_amp, use_fp16 = _resolve_amp(config)
    task_name = _official_task_name(config.dataset_name)
    max_length = TASK_MAX_LENGTH[task_name]

    generator = GeneratorModel(
        GeneratorConfig(
            model_name=config.generator_model_name,
            top_p=0.95,
            top_k=None,
            repetition_penalty=1.0,
            generate_batch_size=512,
            generate_max_length=max_length,
            generate_bf16=config.bf16,
            generate_fp16=config.fp16,
        ),
        num_labels=_num_labels(task_name),
        sentence_keys=get_dataset_info(config.dataset_name).text_columns,
    ).to(device)
    if hasattr(generator.model, "gradient_checkpointing_enable"):
        generator.model.gradient_checkpointing_enable()
    learner = PaperLearnerModel(config.learner_model_name, task_name, _num_labels(task_name)).to(device)
    data = PaperDataModule(config.dataset_name, generator, learner)

    ensure_dir(run.lm_save_dir)
    ensure_dir(run.dataset_dir)
    ensure_dir(run.result_dir)
    _save_json(asdict(config), run.test_save_dir / "config.json")

    _train_lm_phase(generator, data, config, device, amp_dtype, use_amp, use_fp16)
    if config.save_model:
        generator.model.save_pretrained(run.lm_generator_dir / "last-ckpt")
        generator.tokenizer.save_pretrained(run.lm_generator_dir / "tokenizer")

    dataset_list = _generate_dataset_list(generator, data.sentence_keys, config.dpc, config.n_datasets, device, amp_dtype, use_amp)
    for index, dataset in enumerate(dataset_list):
        dataset.to_json(str(run.dataset_dir / f"dataset_{index}.json"))
    results = _evaluate_dataset_list(
        dataset_list,
        data,
        config,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
    )
    _save_json(results, run.result_dir / "results.json")
    _save_json(_average_results(results), run.result_dir / "summary.json")


class PaperLearnerModel(nn.Module):
    def __init__(self, model_name: str, task_name: str, num_labels: int, *, gradient_checkpointing: bool = True):
        super().__init__()
        self.model_name = model_name
        self.num_labels = num_labels
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            finetuning_task=task_name,
            problem_type="single_label_classification",
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
        self.initial_state_dict = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def forward(self, *args, **kwargs) -> SequenceClassifierOutput:
        labels = kwargs.pop("labels", None)
        outputs = self.model(*args, **kwargs)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(outputs.logits.view(-1, self.num_labels), labels.view(-1), reduction="none")
        return SequenceClassifierOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def init_weights(self) -> None:
        self.model.load_state_dict(self.initial_state_dict)
        for module_name in _classifier_module_names(self.model_name):
            module = self.model
            for part in module_name.split("."):
                module = getattr(module, part)
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    submodule.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
                    if submodule.bias is not None:
                        submodule.bias.data.zero_()

    def classifier_param_names(self) -> list[str]:
        out = []
        for module_name in _classifier_module_names(self.model_name):
            module = self.model
            for part in module_name.split("."):
                module = getattr(module, part)
            for name, _ in module.named_parameters():
                out.append(f"model.{module_name}.{name}")
        return out


class PaperDataModule:
    def __init__(self, dataset_name: str, generator: GeneratorModel, learner: PaperLearnerModel):
        self.dataset_name = dataset_name
        self.task_name = _official_task_name(dataset_name)
        self.info = get_dataset_info(dataset_name)
        dataset_dict = load_text_classification_dataset(dataset_name)
        train, eval_dataset = get_train_eval_splits(dataset_dict, dataset_name)
        self.train_dataset = _rename_label_to_labels(train, self.info.label_column)
        self.eval_dataset = _rename_label_to_labels(eval_dataset, self.info.label_column)
        self.sentence_keys = self.info.text_columns
        self.num_labels = len(set(int(x) for x in self.train_dataset["labels"]))
        self.generator = generator
        self.learner = learner
        self.generator_collator = DataCollatorForLanguageModeling(
            tokenizer=generator.tokenizer, mlm=False, pad_to_multiple_of=8
        )
        self.learner_collator = DataCollatorWithPadding(
            tokenizer=learner.tokenizer, padding="longest", pad_to_multiple_of=8
        )

    def preprocess_examples(self, examples: list[dict[str, Any]]) -> list[dict[str, dict[str, Any]]]:
        labels = [int(ex["labels"]) for ex in examples]
        texts_by_key = [[str(ex[key]).strip() for ex in examples] for key in self.sentence_keys]
        joined = [f" {SEP_TOKEN} ".join(parts) for parts in zip(*texts_by_key)]
        generator_texts = [
            f"{self.generator.bos_tokens[label]} {text} {self.generator.tokenizer.eos_token}"
            for label, text in zip(labels, joined)
        ]
        generator_batch = self.generator.tokenizer(
            generator_texts,
            max_length=self.generator.config.generate_max_length,
            truncation=True,
        )
        learner_batch = self.learner.tokenizer(
            *texts_by_key,
            max_length=self.learner.tokenizer.model_max_length,
            truncation=True,
        )
        learner_batch["labels"] = labels
        return [
            {
                "generator": {key: values[i] for key, values in generator_batch.items()},
                "learner": {key: values[i] for key, values in learner_batch.items()},
            }
            for i in range(len(examples))
        ]

    def collate(self, examples: list[dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
        preprocessed = self.preprocess_examples(examples)
        return {
            "generator": self.generator_collator([ex["generator"] for ex in preprocessed]),
            "learner": self.learner_collator([ex["learner"] for ex in preprocessed]),
        }

    def get_train_loader(
        self,
        dataset: Dataset | None = None,
        *,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = True,
        label: int | None = None,
    ) -> DataLoader:
        dataset = dataset if dataset is not None else self.train_dataset
        if label is not None:
            dataset = dataset.filter(lambda ex: int(ex["labels"]) == int(label))
        return DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.collate,
        )

    def eval_loader(self, batch_size: int = 256) -> DataLoader:
        return DataLoader(self.eval_dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=self.collate)


def _run_local_paper_dilm(config: PaperDiLMConfig, run: OfficialDiLMRun) -> None:
    set_seed(config.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = _cuda_device()
    amp_dtype, use_amp, use_fp16 = _resolve_amp(config)
    task_name = _official_task_name(config.dataset_name)
    max_length = TASK_MAX_LENGTH[task_name]
    gm_real_dpc = config.gm_real_dpc if config.gm_real_dpc is not None else _paper_gm_real_dpc(task_name)

    generator = GeneratorModel(
        GeneratorConfig(
            model_name=config.generator_model_name,
            top_p=0.95,
            top_k=None,
            repetition_penalty=1.0,
            generate_batch_size=512,
            generate_max_length=max_length,
            generate_bf16=config.bf16,
            generate_fp16=config.fp16,
        ),
        num_labels=_num_labels(task_name),
        sentence_keys=get_dataset_info(config.dataset_name).text_columns,
    ).to(device)
    if hasattr(generator.model, "gradient_checkpointing_enable"):
        generator.model.gradient_checkpointing_enable()
    learner = PaperLearnerModel(config.learner_model_name, task_name, _num_labels(task_name)).to(device)
    data = PaperDataModule(config.dataset_name, generator, learner)

    ensure_dir(run.lm_save_dir)
    ensure_dir(run.dc_save_dir)
    ensure_dir(run.dataset_dir)
    ensure_dir(run.result_dir)
    _save_json(asdict(config), run.test_save_dir / "config.json")

    _train_lm_phase(generator, data, config, device, amp_dtype, use_amp, use_fp16)
    if config.save_model:
        generator.model.save_pretrained(run.lm_generator_dir / "last-ckpt")
        generator.tokenizer.save_pretrained(run.lm_generator_dir / "tokenizer")

    repset_teachers = _generate_repset_teachers(
        data.train_dataset,
        dpc=gm_real_dpc,
        n=config.n_repset,
        task_name=task_name,
        sentence_keys=data.sentence_keys,
        model_name=config.learner_model_name,
        device=device,
        amp_dtype=amp_dtype,
    )
    _train_dc_phase(
        generator,
        learner,
        data,
        repset_teachers,
        config,
        gm_real_dpc=gm_real_dpc,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
        use_fp16=use_fp16,
    )
    if config.save_model:
        generator.model.save_pretrained(run.dc_generator_dir / "last-ckpt")
        generator.tokenizer.save_pretrained(run.dc_generator_dir / "tokenizer")

    dataset_list = _generate_dataset_list(generator, data.sentence_keys, config.dpc, config.n_datasets, device, amp_dtype, use_amp)
    for index, dataset in enumerate(dataset_list):
        dataset.to_json(str(run.dataset_dir / f"dataset_{index}.json"))
    results = _evaluate_dataset_list(
        dataset_list,
        data,
        config,
        device=device,
        amp_dtype=amp_dtype,
        use_amp=use_amp,
    )
    _save_json(results, run.result_dir / "results.json")
    _save_json(_average_results(results), run.result_dir / "summary.json")


def _train_lm_phase(
    generator: GeneratorModel,
    data: PaperDataModule,
    config: PaperDiLMConfig,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    use_fp16: bool,
) -> None:
    loader = _endless(data.get_train_loader(batch_size=config.lm_batch_size))
    optimizer, scheduler = _configure_optimizer(
        generator,
        lr=config.lm_learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        num_train_steps=config.lm_train_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    generator.train()
    for _ in trange(config.lm_train_steps, desc="DiLM LM phase", leave=False):
        batch = next(loader)["generator"]
        batch = _move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            loss = generator.compute_loss(**batch).mean()
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()


def _train_dc_phase(
    generator: GeneratorModel,
    learner: PaperLearnerModel,
    data: PaperDataModule,
    repset_teachers: list[Dataset],
    config: PaperDiLMConfig,
    *,
    gm_real_dpc: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    use_fp16: bool,
) -> None:
    assert config.dc_train_steps % config.inner_loop == 0
    num_labels = data.num_labels
    optimizer, scheduler = _configure_optimizer(
        generator,
        lr=config.dc_learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.05,
        num_train_steps=config.dc_train_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    params = {k: v.detach() for k, v in learner.named_parameters()}
    buffers = {k: v for k, v in learner.named_buffers()}
    classifier_param_names = learner.classifier_param_names()
    gm_real_loaders = _gm_real_loaders(data, repset_teachers, gm_real_dpc, num_labels)

    outer_loop = config.dc_train_steps // config.inner_loop
    gm_syn_loaders: dict[int, Iterator[dict[str, dict[str, torch.Tensor]]]] | None = None
    learner_train_loader = _endless(data.get_train_loader(batch_size=64))

    for outer in trange(outer_loop, desc="DiLM DC outer loop", leave=False):
        if outer % config.generate_dataset_interval == 0 or gm_syn_loaders is None:
            gm_syn_loaders = _gm_syn_loaders(
                generator,
                learner,
                data,
                config.gm_syn_dpc,
                num_datasets=config.inner_loop * config.generate_dataset_interval,
                device=device,
                amp_dtype=amp_dtype,
                use_amp=use_amp,
            )

        learner.init_weights()
        learner.to(device)
        learner_optimizer, learner_scheduler = _configure_optimizer(
            learner,
            lr=1e-4,
            weight_decay=0.01,
            warmup_ratio=0.5,
            num_train_steps=config.inner_loop * config.model_step_per_inner_step,
        )
        learner_scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
        generator.train()

        for inner in range(config.inner_loop):
            optimizer.zero_grad(set_to_none=True)
            grad_sim_total = 0.0
            for label in range(num_labels):
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    with torch.no_grad():
                        real_batch = next(gm_real_loaders[label])["learner"]
                        real_inputs = _move_batch(real_batch, device)
                        grad_real = _compute_grad(
                            learner,
                            params,
                            buffers,
                            real_inputs,
                            classifier_param_names=classifier_param_names,
                        ).detach()

                    syn_batch = next(gm_syn_loaders[label])
                    gen_inputs = _move_batch(syn_batch["generator"], device)
                    gen_losses = generator.compute_loss(**gen_inputs)
                    loss_weights = F.softmax(-gen_losses, dim=-1)
                    syn_inputs = _move_batch(syn_batch["learner"], device)
                    grad_syn = _compute_grad(
                        learner,
                        params,
                        buffers,
                        syn_inputs,
                        classifier_param_names=classifier_param_names,
                        loss_weights=loss_weights,
                    )
                    grad_sim = F.cosine_similarity(grad_real, grad_syn, dim=0)
                    loss = (1.0 - grad_sim) / num_labels
                scaler.scale(loss).backward()
                grad_sim_total += float(grad_sim.detach().cpu())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if inner + 1 < config.inner_loop:
                for _ in range(config.model_step_per_inner_step):
                    batch = _move_batch(next(learner_train_loader)["learner"], device)
                    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                        loss_learner = learner(**batch).loss.mean()
                    learner_optimizer.zero_grad(set_to_none=True)
                    learner_scaler.scale(loss_learner).backward()
                    learner_scaler.unscale_(learner_optimizer)
                    torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
                    learner_scaler.step(learner_optimizer)
                    learner_scaler.update()
                    learner_scheduler.step()


def _compute_grad(
    learner: PaperLearnerModel,
    params: dict[str, torch.Tensor],
    buffers: dict[str, torch.Tensor],
    inputs: dict[str, torch.Tensor],
    *,
    classifier_param_names: list[str],
    loss_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    def loss_fn(p, b):
        outputs = torch.func.functional_call(learner, (p, b), kwargs=inputs)
        losses = outputs.loss
        if loss_weights is None:
            return losses.mean()
        return losses.dot(loss_weights)

    grads = torch.func.grad(loss_fn)(params, buffers)
    return torch.cat([grads[name].reshape(-1) for name in classifier_param_names], dim=0)


def _generate_dataset_list(
    generator: GeneratorModel,
    sentence_keys: tuple[str, ...],
    dpc: int,
    n: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> list[Dataset]:
    num_labels = generator.num_labels
    total = dpc * num_labels * n
    data_size_per_dataset = dpc * num_labels
    pending = {
        sample_id: {
            "sample_id": sample_id,
            "label": sample_id % num_labels,
        }
        for sample_id in range(total)
    }
    generated: dict[int, dict[str, Any]] = {}
    retries = 0
    with tqdm(total=total, desc="Generating DiLM data", leave=False) as pbar:
        while pending:
            batch_ids = list(pending)[: generator.config.generate_batch_size]
            batch_labels = [pending[sample_id]["label"] for sample_id in batch_ids]
            prompt = torch.tensor([[generator.bos_ids[label]] for label in batch_labels], dtype=torch.long, device=device)
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                out = generator.model.generate(
                    prompt,
                    do_sample=True,
                    top_p=generator.config.top_p,
                    top_k=generator.config.top_k,
                    repetition_penalty=generator.config.repetition_penalty,
                    max_length=generator.config.generate_max_length,
                    pad_token_id=generator.tokenizer.eos_token_id,
                    bad_words_ids=[[bos_id] for bos_id in generator.bos_ids.values()],
                )
            texts = generator.tokenizer.batch_decode(out[:, 1:], skip_special_tokens=True)
            good = 0
            for sample_id, label, text in zip(batch_ids, batch_labels, texts):
                parts = [part.strip() for part in text.split(SEP_TOKEN)]
                if len(parts) >= len(sentence_keys) and all(part != "" for part in parts[: len(sentence_keys)]):
                    example = {key: part for key, part in zip(sentence_keys, parts[: len(sentence_keys)])}
                    example["labels"] = label
                    generated[sample_id] = example
                    pending.pop(sample_id)
                    good += 1
            retries += len(batch_ids) - good
            if retries > total:
                raise RuntimeError("Too many DiLM generation failures.")
            pbar.update(good)

    ordered = [example for _, example in sorted(generated.items())]
    return [
        Dataset.from_list(ordered[data_size_per_dataset * i : data_size_per_dataset * (i + 1)])
        for i in range(n)
    ]


def _gm_real_loaders(data: PaperDataModule, repset_teachers: list[Dataset], batch_size: int, num_labels: int):
    concat = concatenate_datasets(repset_teachers)
    return {
        label: _endless(data.get_train_loader(concat, batch_size=batch_size, shuffle=False, drop_last=False, label=label))
        for label in range(num_labels)
    }


def _gm_syn_loaders(
    generator: GeneratorModel,
    learner: PaperLearnerModel,
    data: PaperDataModule,
    dpc: int,
    *,
    num_datasets: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
):
    synthetic = concatenate_datasets(
        _generate_dataset_list(generator, data.sentence_keys, dpc, num_datasets, device, amp_dtype, use_amp)
    )
    return _cluster_wise_loaders(
        synthetic,
        data,
        learner,
        dpc=dpc,
        n_clusters=dpc,
        max_iteration=num_datasets,
        device=device,
        amp_dtype=amp_dtype,
    )


def _cluster_wise_loaders(
    dataset: Dataset,
    data: PaperDataModule,
    learner: PaperLearnerModel,
    *,
    dpc: int,
    n_clusters: int,
    max_iteration: int,
    device: torch.device,
    amp_dtype: torch.dtype,
):
    loaders = {}
    for label in range(data.num_labels):
        label_dataset = dataset.filter(lambda ex: int(ex["labels"]) == label)
        cluster_ids = _cluster_dataset(label_dataset, data, learner, n_clusters, device, amp_dtype)
        clustered = label_dataset.add_column("cluster_id", cluster_ids)
        cluster_datasets = [clustered.filter(lambda ex, cid=cid: ex["cluster_id"] == cid).remove_columns("cluster_id") for cid in range(n_clusters)]
        batch_size_per_cluster = max(1, dpc // n_clusters)
        queued = []
        for idx in range(batch_size_per_cluster * max_iteration):
            for cluster_dataset in cluster_datasets:
                queued.append(cluster_dataset[idx % len(cluster_dataset)])
        loaders[label] = iter(DataLoader(Dataset.from_list(queued), batch_size=dpc, shuffle=False, drop_last=False, collate_fn=data.collate))
    return loaders


def _cluster_dataset(dataset: Dataset, data: PaperDataModule, learner: PaperLearnerModel, n_clusters: int, device: torch.device, amp_dtype: torch.dtype) -> list[int]:
    embeddings = _learner_embeddings(dataset, data, learner, device, amp_dtype, batch_size=512)
    return KMeans(n_clusters=n_clusters, init="k-means++", max_iter=300, n_init=1).fit_predict(embeddings.cpu()).tolist()


def _generate_repset_teachers(
    dataset: Dataset,
    *,
    dpc: int,
    n: int,
    task_name: str,
    sentence_keys: tuple[str, ...],
    model_name: str,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> list[Dataset]:
    return [_k_centers_dataset(dataset, dpc, task_name, sentence_keys, model_name, seed=i, device=device, amp_dtype=amp_dtype) for i in range(n)]


def _k_centers_dataset(dataset: Dataset, dpc: int, task_name: str, sentence_keys: tuple[str, ...], model_name: str, seed: int, device: torch.device, amp_dtype: torch.dtype) -> Dataset:
    parts = []
    for label in range(_num_labels(task_name)):
        label_dataset = dataset.filter(lambda ex: int(ex["labels"]) == label)
        embeddings = _encoder_embeddings(label_dataset, sentence_keys, model_name, device, amp_dtype)
        ids = _kmeans_center_ids(embeddings, dpc, seed)
        parts.append(label_dataset.select(ids))
    return concatenate_datasets(parts)


def _encoder_embeddings(dataset: Dataset, sentence_keys: tuple[str, ...], model_name: str, device: torch.device, amp_dtype: torch.dtype) -> torch.Tensor:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    outputs = []
    for start in range(0, len(dataset), 256):
        batch = dataset[start : start + 256]
        texts = tuple(batch[key] for key in sentence_keys)
        enc = tokenizer(*texts, padding=True, truncation=True, return_tensors="pt")
        enc = _move_batch(enc, device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            hidden = model(**enc, output_hidden_states=True).hidden_states[-1][:, 0].detach().cpu()
        outputs.append(hidden)
    del model
    torch.cuda.empty_cache()
    return torch.cat(outputs, dim=0)


def _learner_embeddings(dataset: Dataset, data: PaperDataModule, learner: PaperLearnerModel, device: torch.device, amp_dtype: torch.dtype, batch_size: int) -> torch.Tensor:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=data.collate)
    embeddings = []
    learner.eval()
    for batch in loader:
        inputs = _move_batch(batch["learner"], device)
        inputs.pop("labels", None)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            hidden = learner.model(**inputs, output_hidden_states=True).hidden_states[-1][:, 0].detach().cpu()
        embeddings.append(hidden)
    return torch.cat(embeddings, dim=0)


def _kmeans_center_ids(embeddings: torch.Tensor, k: int, seed: int) -> list[int]:
    kmeans = KMeans(n_clusters=k, init="k-means++", max_iter=300, n_init=1, random_state=seed)
    cluster_ids = kmeans.fit_predict(embeddings.cpu()).tolist()
    ids = []
    for cid in range(k):
        sample_ids = [i for i, cluster_id in enumerate(cluster_ids) if cluster_id == cid]
        cluster = embeddings[sample_ids]
        centroid = cluster.mean(dim=0)
        dists = (cluster - centroid.unsqueeze(0)).pow(2).sum(dim=1).sqrt()
        ids.append(sample_ids[int(torch.argmin(dists))])
    return ids


def _evaluate_dataset_list(
    datasets: list[Dataset],
    data: PaperDataModule,
    config: PaperDiLMConfig,
    *,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
) -> list[dict[str, float]]:
    results = []
    for dataset in datasets:
        for _ in range(config.n_eval_per_dataset):
            learner = PaperLearnerModel(config.learner_model_name, data.task_name, data.num_labels).to(device)
            learner.init_weights()
            loader = _endless(data.get_train_loader(dataset, batch_size=config.eval_batch_size, shuffle=True, drop_last=True))
            optimizer, scheduler = _configure_optimizer(
                learner,
                lr=1e-4,
                weight_decay=0.01,
                warmup_ratio=0.5,
                num_train_steps=config.eval_train_steps,
            )
            for _step in range(config.eval_train_steps):
                batch = _move_batch(next(loader)["learner"], device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    loss = learner(**batch).loss.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(learner.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            results.append(_evaluate_learner(learner, data, device, amp_dtype, use_amp))
            del learner
            torch.cuda.empty_cache()
    return results


def _evaluate_learner(learner: PaperLearnerModel, data: PaperDataModule, device: torch.device, amp_dtype: torch.dtype, use_amp: bool) -> dict[str, float]:
    learner.eval()
    preds = []
    labels = []
    losses = []
    for batch in data.eval_loader():
        inputs = _move_batch(batch["learner"], device)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            outputs = learner(**inputs)
        preds.extend(outputs.logits.argmax(dim=-1).detach().cpu().tolist())
        labels.extend(inputs["labels"].detach().cpu().tolist())
        losses.extend(outputs.loss.detach().cpu().tolist())
    metrics = compute_classification_metrics(labels, preds, metric_name=data.info.metric_name)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def _configure_optimizer(model: nn.Module, *, lr: float, weight_decay: float, warmup_ratio: float, num_train_steps: int):
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(grouped, lr=lr)
    scheduler = get_scheduler("cosine", optimizer=optimizer, num_warmup_steps=math.floor(num_train_steps * warmup_ratio), num_training_steps=num_train_steps)
    return optimizer, scheduler


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _endless(loader: DataLoader) -> Iterator[Any]:
    while True:
        yield from loader


def _average_results(results: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = results[0].keys()
    return {key: {"mean": float(np.mean([r[key] for r in results])), "std": float(np.std([r[key] for r in results]))} for key in keys}


def _rename_label_to_labels(dataset: Dataset, label_column: str) -> Dataset:
    if label_column != "labels" and "labels" not in dataset.column_names:
        return dataset.rename_column(label_column, "labels")
    return dataset


def _classifier_module_names(model_name: str) -> list[str]:
    if model_name.startswith("xlnet"):
        return ["sequence_summary", "logits_proj"]
    if model_name.startswith("gpt2"):
        return ["score"]
    return ["classifier"]


def _official_task_name(dataset_name: str) -> str:
    name = get_dataset_info(dataset_name).name
    if name in {"sst2", "qqp", "mnli"}:
        return name
    if name == "mnli-m":
        return "mnli"
    raise ValueError("Paper DiLM supports only sst2, qqp, and mnli-m.")


def _paper_gm_real_dpc(task_name: str) -> int:
    return 200 if task_name == "sst2" else 100


def _num_labels(task_name: str) -> int:
    return {"sst2": 2, "qqp": 2, "mnli": 3}[task_name]


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Paper DiLM port requires CUDA. Run on V100/A100.")
    return torch.device("cuda")


def _resolve_amp(config: PaperDiLMConfig) -> tuple[torch.dtype, bool, bool]:
    if config.fp16:
        return torch.float16, True, True
    if config.bf16:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, True, False
        print("[DiLM] bf16 requested but this CUDA device does not support bf16; using fp16.")
        return torch.float16, True, True
    return torch.float32, False, False


def _has_generated_datasets(dataset_dir: Path, n_datasets: int) -> bool:
    return all((dataset_dir / f"dataset_{index}.json").exists() for index in range(n_datasets))


def _save_json(data: Any, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, default=str)
        file.write("\n")
