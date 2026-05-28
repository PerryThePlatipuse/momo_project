import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset, DatasetDict, disable_progress_bar, load_dataset, load_from_disk
from torch.cuda import amp
from torch.nn import functional as F
from torch.optim import SGD, Adam, AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from transformers import DataCollatorWithPadding, get_scheduler

from dataset_attrs import DATASET_ATTRS
from learner import LearnerConfig, LearnerModel

from .embedding_data import (
    EmbeddingDistillationDataConfig,
    EmbeddingLearnerTrainConfig,
    SyntheticEmbeddingDataset,
)

logger = logging.getLogger(__name__)
disable_progress_bar()
MISSING = object()


@dataclass
class EmbeddingDistillationDataModuleConfig:
    task_name: str
    datasets_path: str | os.PathLike
    preprocessed_datasets_path: str | os.PathLike
    train_batch_size: int = 32
    eval_batch_size: int = 128
    num_proc: int = 1
    force_preprocess: bool = False
    max_length: int | None = None


class EmbeddingDistillationDataModule:
    """Learner-only data module for embedding-level distillation."""

    def __init__(self, config: EmbeddingDistillationDataModuleConfig, learner: LearnerModel):
        self.config = config
        self.learner = learner
        self.dataset_attr = DATASET_ATTRS[config.task_name]
        self.num_labels = self.dataset_attr["num_labels"]
        self.datasets = self._load_raw_dataset()
        self.preprocessed_datasets = self._load_or_preprocess()
        self.data_collator = DataCollatorWithPadding(
            tokenizer=learner.tokenizer,
            padding="longest",
            pad_to_multiple_of=8,
        )

    @property
    def max_length(self) -> int:
        if self.config.max_length is not None:
            return self.config.max_length
        return min(self.dataset_attr["max_length"], self.learner.tokenizer.model_max_length)

    def _load_raw_dataset(self) -> DatasetDict:
        if os.path.exists(self.config.datasets_path):
            datasets = load_from_disk(self.config.datasets_path)
        else:
            datasets = load_dataset(*self.dataset_attr["load_args"])
            if "validation" not in datasets:
                datasets["validation"] = datasets.pop(self.dataset_attr["test_split_key"])
            Path(self.config.datasets_path).parent.mkdir(parents=True, exist_ok=True)
            datasets.save_to_disk(self.config.datasets_path)

        if self.dataset_attr["label_key"] != "labels":
            datasets = datasets.rename_column(self.dataset_attr["label_key"], "labels")
        return datasets

    def _load_or_preprocess(self) -> DatasetDict:
        if os.path.exists(self.config.preprocessed_datasets_path) and not self.config.force_preprocess:
            return load_from_disk(self.config.preprocessed_datasets_path)

        datasets = self.preprocess_dataset(self.datasets)
        Path(self.config.preprocessed_datasets_path).parent.mkdir(parents=True, exist_ok=True)
        datasets.save_to_disk(self.config.preprocessed_datasets_path)
        return datasets

    def preprocess_dataset(self, dataset: Dataset | DatasetDict) -> Dataset | DatasetDict:
        sentence_keys = self.dataset_attr["sentence_keys"]
        tokenizer = self.learner.tokenizer
        num_proc = self.config.num_proc if self.config.num_proc > 1 else None

        def tokenize(batch: dict[str, Any]) -> dict[str, Any]:
            batch_sentences = [[s.strip() for s in batch[key]] for key in sentence_keys]
            encoded = tokenizer(
                *batch_sentences,
                max_length=self.max_length,
                truncation=True,
            )
            encoded["labels"] = batch["labels"]
            return encoded

        map_kwargs = {"batched": True, "desc": "Tokenize datasets"}
        if num_proc is not None:
            map_kwargs["num_proc"] = num_proc
        dataset = dataset.map(tokenize, **map_kwargs)

        if isinstance(dataset, Dataset):
            column_names = dataset.column_names
        else:
            column_names = dataset["train"].column_names
        removed = [col for col in column_names if col not in {"input_ids", "attention_mask", "token_type_ids", "labels"}]
        return dataset.remove_columns(removed)

    def train_loader(
        self,
        *,
        batch_size: int | None = None,
        shuffle: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        return DataLoader(
            self.preprocessed_datasets["train"],
            batch_size=batch_size or self.config.train_batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            collate_fn=self.data_collator,
        )

    def eval_loader(self) -> DataLoader:
        return DataLoader(
            self.preprocessed_datasets["validation"],
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=self.data_collator,
        )

    def balanced_real_input_embeddings(
        self,
        learner: LearnerModel,
        *,
        dpc: int,
        seq_length: int,
        device: torch.device,
    ) -> torch.Tensor:
        tokenizer = learner.tokenizer
        sentence_keys = self.dataset_attr["sentence_keys"]
        by_label: dict[int, list[dict[str, Any]]] = {label: [] for label in range(self.num_labels)}
        for row in self.datasets["train"]:
            label = row["labels"]
            if len(by_label[label]) < dpc:
                by_label[label].append(row)
            if all(len(rows) >= dpc for rows in by_label.values()):
                break

        rows = []
        for example_idx in range(dpc):
            for label in range(self.num_labels):
                if example_idx >= len(by_label[label]):
                    raise ValueError(f"Dataset has fewer than {dpc} examples for label {label}.")
                row = by_label[label][example_idx]
                rows.append([row[key].strip() for key in sentence_keys])

        batch_sentences = list(zip(*rows))
        encoded = tokenizer(
            *batch_sentences,
            padding="max_length",
            max_length=seq_length,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            return learner.get_input_embeddings()(encoded["input_ids"]).detach().cpu()


@dataclass
class EmbeddingDistillationTrainConfig:
    epoch: int = 3
    lr_input_embeds: float = 1.0e-2
    lr_attention_labels: float = 1.0e-2
    lr_labels: float = 1.0e-3
    lr_lrs: float = 1.0e-2
    optimizer_type: str = "adamw"
    scheduler_type: str = "linear"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.0
    max_grad_norm: float | None = 1.0
    val_interval: int = 1
    log_interval: int = 10
    n_eval_model: int = 3
    max_train_batches_per_epoch: int | None = None
    max_eval_batches: int | None = None
    save_dir: str = "results/embedding_distillation/run"
    fp16: bool = False
    bf16: bool = False

    def __post_init__(self):
        assert not (self.fp16 and self.bf16)


class EmbeddingDistillationTrainer:
    def __init__(self, config: EmbeddingDistillationTrainConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(
        self,
        synthetic_data: SyntheticEmbeddingDataset,
        learner: LearnerModel,
        train_loader: DataLoader,
        eval_loader: DataLoader,
    ) -> dict[str, float]:
        learner.to(self.device)
        synthetic_data.to(self.device)

        steps_per_epoch = len(train_loader)
        if self.config.max_train_batches_per_epoch is not None:
            steps_per_epoch = min(steps_per_epoch, self.config.max_train_batches_per_epoch)
        total_steps = self.config.epoch * steps_per_epoch

        optimizer = self._configure_optimizer(synthetic_data)
        scheduler = get_scheduler(
            name=self.config.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * self.config.warmup_ratio),
            num_training_steps=total_steps,
        )
        scaler = amp.GradScaler(enabled=self.use_amp)
        evaluator = EmbeddingDistillationEvaluator(
            task_name=learner.bert_model_config.finetuning_task,
            num_labels=synthetic_data.num_labels,
            device=self.device,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            max_eval_batches=self.config.max_eval_batches,
        )

        save_dir = Path(self.config.save_dir)
        best_dir = save_dir / "best-ckpt"
        last_dir = save_dir / "last-ckpt"
        save_dir.mkdir(parents=True, exist_ok=True)
        self._save_run_config(synthetic_data, save_dir)

        best_loss = float("inf")
        best_results: dict[str, float] = {}

        for epoch in trange(self.config.epoch, dynamic_ncols=True, desc="Distill embeddings"):
            train_loss = 0.0
            learner.train()
            for step, real_batch in enumerate(train_loader):
                if self.config.max_train_batches_per_epoch is not None and step >= self.config.max_train_batches_per_epoch:
                    break
                loss = self._outer_loss(synthetic_data, learner, real_batch)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                if self.config.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(synthetic_data.parameters(), self.config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss += loss.item()
                if self.config.log_interval > 0 and (step + 1) % self.config.log_interval == 0:
                    logger.info(
                        "epoch=%s step=%s train_loss=%.4f",
                        epoch + 1,
                        step + 1,
                        train_loss / self.config.log_interval,
                    )
                    train_loss = 0.0

            if (epoch + 1) % self.config.val_interval == 0:
                results = evaluator.evaluate(
                    synthetic_data,
                    learner,
                    eval_loader,
                    n_eval_model=self.config.n_eval_model,
                )
                logger.info("validation epoch=%s results=%s", epoch + 1, results)
                if results["loss"] < best_loss:
                    best_loss = results["loss"]
                    best_results = results
                    synthetic_data.save_pretrained(best_dir)

        synthetic_data.save_pretrained(last_dir)
        if best_dir.exists():
            best = SyntheticEmbeddingDataset.from_pretrained(best_dir)
            synthetic_data.load_data_dict(best.data_dict(detach=True))
        return best_results

    def _outer_loss(
        self,
        synthetic_data: SyntheticEmbeddingDataset,
        learner: LearnerModel,
        real_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        learner.init_weights()
        learner.train()

        params = {
            name: value.detach().requires_grad_(value.requires_grad)
            for name, value in learner.named_parameters()
        }
        buffers = {name: value for name, value in learner.named_buffers()}

        for inner_step in range(synthetic_data.train_config.inner_steps):
            syn_batch = synthetic_data.get_batch(inner_step)
            syn_lr = syn_batch.pop("lr")
            loss_kwargs = {
                "inputs_embeds": syn_batch["inputs_embeds"],
                "labels": syn_batch["labels"],
                "attention_mask": torch.ones(
                    syn_batch["inputs_embeds"].shape[:2],
                    dtype=torch.long,
                    device=self.device,
                ),
                "attention_labels": syn_batch["attention_labels"],
                "attention_loss_lambda": synthetic_data.attention_loss_lambda,
            }
            grads = torch.func.grad(self._functional_loss)(params, buffers, learner, loss_kwargs)
            params = {name: param - syn_lr * grads[name] for name, param in params.items()}

        real_batch = batch_to_device(real_batch, self.device)
        return self._functional_loss(params, buffers, learner, real_batch)

    def _functional_loss(
        self,
        params: dict[str, torch.Tensor],
        buffers: dict[str, torch.Tensor],
        learner: LearnerModel,
        batch: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        batch = dict(batch)
        attention_labels = batch.pop("attention_labels", None)
        attention_loss_lambda = batch.pop("attention_loss_lambda", 1.0)
        batch["output_attentions"] = attention_labels is not None

        with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
            outputs = torch.func.functional_call(learner, (params, buffers), args=(), kwargs=batch)
            loss_task = outputs.loss.mean()

            if attention_labels is None:
                return loss_task

            attn_weights = torch.stack(outputs.attentions, dim=1)
            attn_weights = attn_weights[..., : attention_labels.size(-2), :]
            if attn_weights.shape != attention_labels.shape:
                raise ValueError(f"Attention shape mismatch: {attn_weights.shape} != {attention_labels.shape}")
            loss_attn = F.kl_div(
                torch.log(attn_weights + 1e-12),
                attention_labels,
                reduction="none",
            )
            loss_attn = loss_attn.sum(-1).mean()
            return loss_task + attention_loss_lambda * loss_attn

    def _configure_optimizer(self, synthetic_data: SyntheticEmbeddingDataset):
        optimizer_class = {"sgd": SGD, "adam": Adam, "adamw": AdamW}
        if self.config.optimizer_type not in optimizer_class:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_type}")
        return optimizer_class[self.config.optimizer_type](
            synthetic_data.parameter_groups(
                lr_input_embeds=self.config.lr_input_embeds,
                lr_labels=self.config.lr_labels,
                lr_lrs=self.config.lr_lrs,
                lr_attention_labels=self.config.lr_attention_labels,
                weight_decay=self.config.weight_decay,
            )
        )

    def _save_run_config(self, synthetic_data: SyntheticEmbeddingDataset, save_dir: Path):
        with open(save_dir / "run_config.json", "w") as f:
            json.dump(
                {
                    "train": asdict(self.config),
                    "synthetic_data": asdict(synthetic_data.config),
                    "learner_train": asdict(synthetic_data.train_config),
                },
                f,
                indent=2,
            )

    @property
    def use_amp(self) -> bool:
        return (self.config.fp16 or self.config.bf16) and self.device.type == "cuda"

    @property
    def amp_dtype(self):
        return torch.float16 if self.config.fp16 else torch.bfloat16


class EmbeddingDistillationEvaluator:
    def __init__(
        self,
        *,
        task_name: str,
        num_labels: int,
        device: torch.device | str | None = None,
        fp16: bool = False,
        bf16: bool = False,
        max_eval_batches: int | None = None,
    ):
        self.task_name = task_name
        self.num_labels = num_labels
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fp16 = fp16
        self.bf16 = bf16
        self.max_eval_batches = max_eval_batches

    def evaluate(
        self,
        synthetic_data: SyntheticEmbeddingDataset,
        learner: LearnerModel,
        eval_loader: DataLoader,
        *,
        n_eval_model: int = 3,
    ) -> dict[str, float]:
        learner.to(self.device)
        synthetic_data.to(self.device)
        all_results = []
        for _ in trange(n_eval_model, dynamic_ncols=True, leave=False, desc="Evaluate synthetic data"):
            learner.init_weights()
            self.train_learner(learner, synthetic_data)
            all_results.append(self.evaluate_learner(learner, eval_loader))
        return average_dicts(all_results)

    def train_learner(self, learner: LearnerModel, synthetic_data: SyntheticEmbeddingDataset):
        learner.train()
        for step in range(synthetic_data.train_config.inner_steps):
            batch = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in synthetic_data.get_batch(step).items()
            }
            attention_labels = batch["attention_labels"]
            outputs = learner(
                inputs_embeds=batch["inputs_embeds"],
                labels=batch["labels"],
                attention_mask=torch.ones(
                    batch["inputs_embeds"].shape[:2],
                    dtype=torch.long,
                    device=self.device,
                ),
                output_attentions=attention_labels is not None,
            )
            loss = outputs.loss.mean()
            if attention_labels is not None:
                attn_weights = torch.stack(outputs.attentions, dim=1)
                attn_weights = attn_weights[..., : attention_labels.size(-2), :]
                loss_attn = F.kl_div(
                    torch.log(attn_weights + 1e-12),
                    attention_labels,
                    reduction="none",
                )
                loss = loss + synthetic_data.attention_loss_lambda * loss_attn.sum(-1).mean()

            learner.zero_grad(set_to_none=True)
            loss.backward()
            with torch.no_grad():
                for param in learner.parameters():
                    if param.grad is not None:
                        param.sub_(batch["lr"] * param.grad)

    @torch.inference_mode()
    def evaluate_learner(self, learner: LearnerModel, eval_loader: DataLoader) -> dict[str, float]:
        learner.eval()
        logits, labels = [], []
        total_loss, num_samples = 0.0, 0
        for batch_idx, batch in enumerate(tqdm(eval_loader, dynamic_ncols=True, leave=False, desc="Evaluate learner")):
            if self.max_eval_batches is not None and batch_idx >= self.max_eval_batches:
                break
            batch = batch_to_device(batch, self.device)
            with amp.autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = learner(**batch)
            logits.append(outputs.logits.detach().cpu())
            labels.append(batch["labels"].detach().cpu())
            total_loss += outputs.loss.sum().item()
            num_samples += len(batch["labels"])

        logits_tensor = torch.cat(logits, dim=0)
        labels_tensor = torch.cat(labels, dim=0)
        results = classification_metrics(logits_tensor, labels_tensor, self.num_labels)
        results["loss"] = total_loss / num_samples
        return results

    @property
    def use_amp(self) -> bool:
        return (self.fp16 or self.bf16) and self.device.type == "cuda"

    @property
    def amp_dtype(self):
        return torch.float16 if self.fp16 else torch.bfloat16


def build_synthetic_embedding_dataset(
    *,
    data_config: EmbeddingDistillationDataConfig,
    train_config: EmbeddingLearnerTrainConfig,
    learner: LearnerModel,
    num_labels: int,
) -> SyntheticEmbeddingDataset:
    model_config = learner.bert_model_config
    return SyntheticEmbeddingDataset(
        config=data_config,
        train_config=train_config,
        num_labels=num_labels,
        hidden_size=get_config_value(model_config, "hidden_size", "d_model", "n_embd"),
        num_layers=get_config_value(
            model_config,
            "num_hidden_layers",
            "n_layer",
            "num_layers",
            default=None,
        ),
        num_heads=get_config_value(
            model_config,
            "num_attention_heads",
            "n_head",
            "num_heads",
            default=None,
        ),
    )


def run_embedding_distillation(
    *,
    task_name: str,
    model_name: str = "bert-base-uncased",
    dpc: int = 1,
    seq_length: int = 128,
    inner_steps: int = 1,
    train_epochs: int = 1,
    batch_size_per_label: int | None = None,
    label_type: str = "hard",
    attention_label_type: str = "none",
    save_dir: str | os.PathLike = "results/embedding_distillation/run",
    data_root: str | os.PathLike = "data",
    train_batch_size: int = 32,
    eval_batch_size: int = 128,
    max_train_batches_per_epoch: int | None = None,
    max_eval_batches: int | None = None,
    n_eval_model: int = 3,
    seed: int = 42,
    bf16: bool = False,
    init_strategy: str = "random",
) -> tuple[SyntheticEmbeddingDataset, dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    learner = LearnerModel(
        LearnerConfig(model_name=model_name, disable_dropout=True),
        task_name=task_name,
    )
    data_root = Path(data_root)
    data_module = EmbeddingDistillationDataModule(
        EmbeddingDistillationDataModuleConfig(
            task_name=task_name,
            datasets_path=data_root / task_name / "datasets",
            preprocessed_datasets_path=data_root / task_name / f"embedding_{model_name}",
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            max_length=seq_length,
        ),
        learner=learner,
    )
    data_cfg = EmbeddingDistillationDataConfig(
        dpc=dpc,
        seq_length=seq_length,
        label_type=label_type,
        attention_label_type=attention_label_type,
        init_strategy=init_strategy,
    )
    learner_train_cfg = EmbeddingLearnerTrainConfig(
        inner_steps=inner_steps,
        batch_size_per_label=batch_size_per_label or dpc,
    )
    synthetic_data = build_synthetic_embedding_dataset(
        data_config=data_cfg,
        train_config=learner_train_cfg,
        learner=learner,
        num_labels=data_module.num_labels,
    )
    if data_cfg.init_strategy == "real":
        init_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        learner.to(init_device)
        init_embeds = data_module.balanced_real_input_embeddings(
            learner,
            dpc=dpc,
            seq_length=seq_length,
            device=init_device,
        )
        synthetic_data.initialize_embeddings(init_embeds)

    trainer = EmbeddingDistillationTrainer(
        EmbeddingDistillationTrainConfig(
            epoch=train_epochs,
            save_dir=str(save_dir),
            n_eval_model=n_eval_model,
            max_train_batches_per_epoch=max_train_batches_per_epoch,
            max_eval_batches=max_eval_batches,
            bf16=bf16,
        )
    )
    results = trainer.fit(
        synthetic_data,
        learner,
        data_module.train_loader(),
        data_module.eval_loader(),
    )
    return synthetic_data, results


def classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_labels: int,
) -> dict[str, float]:
    preds = logits.argmax(dim=-1)
    refs = labels
    f1_scores = []
    for label in range(num_labels):
        true_positive = ((preds == label) & (refs == label)).sum().item()
        false_positive = ((preds == label) & (refs != label)).sum().item()
        false_negative = ((preds != label) & (refs == label)).sum().item()
        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_scores.append(f1)

    return {
        "accuracy": (preds == refs).float().mean().item(),
        "macro_f1": float(np.mean(f1_scores)),
    }


def average_dicts(results: list[dict[str, float]]) -> dict[str, float]:
    return {key: float(np.mean([result[key] for result in results])) for key in results[0]}


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def get_config_value(config, *names: str, default=MISSING):
    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return value
    if default is not MISSING:
        return default
    raise AttributeError(f"None of {names} found in {type(config).__name__}.")
