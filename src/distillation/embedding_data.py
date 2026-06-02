import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch.nn import functional as F

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingDistillationDataConfig:
    dpc: int = 1
    seq_length: int = 128
    label_type: Literal["hard", "soft"] = "hard"
    attention_label_type: Literal["none", "cls", "all"] = "none"
    attention_loss_lambda: float = 1.0
    lr_init: float = 1.0e-2
    learn_inner_lrs: bool = True
    lr_linear_decay: bool = False
    fix_order: bool = True
    init_strategy: Literal["random", "real"] = "random"

    def __post_init__(self):
        if self.learn_inner_lrs and self.lr_linear_decay:
            logger.warning("`lr_linear_decay=True` is ignored when learning per-step LR.")


@dataclass
class EmbeddingLearnerTrainConfig:
    inner_steps: int = 1
    batch_size_per_label: int = 1


class SyntheticEmbeddingDataset:
    """Trainable synthetic dataset in BERT embedding space."""

    def __init__(
        self,
        config: EmbeddingDistillationDataConfig,
        train_config: EmbeddingLearnerTrainConfig,
        num_labels: int,
        hidden_size: int,
        num_layers: int | None = None,
        num_heads: int | None = None,
    ):
        self.config = config
        self.train_config = train_config
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads

        if self.config.fix_order:
            assert config.dpc % train_config.batch_size_per_label == 0

        n_examples = config.dpc * num_labels
        self.input_embeds = torch.randn(
            n_examples,
            config.seq_length,
            hidden_size,
            requires_grad=True,
        )
        self.label_ids = torch.arange(n_examples, dtype=torch.long) % num_labels

        self.label_logits = None
        if config.label_type == "soft":
            self.label_logits = torch.eye(num_labels)[self.label_ids].requires_grad_()

        self.lr_logits = self._inverse_softplus(torch.tensor(config.lr_init))
        if config.learn_inner_lrs:
            self.lr_logits = (
                self.lr_logits.unsqueeze(0)
                .expand(train_config.inner_steps)
                .clone()
                .requires_grad_()
            )
        else:
            self.lr_logits = self.lr_logits.requires_grad_()

        self.attention_logits = None
        if config.attention_label_type in {"cls", "all"}:
            if num_layers is None or num_heads is None:
                raise ValueError("num_layers and num_heads are required for attention labels.")
            query_len = 1 if config.attention_label_type == "cls" else config.seq_length
            self.attention_logits = torch.randn(
                n_examples,
                num_layers,
                num_heads,
                query_len,
                config.seq_length,
                requires_grad=True,
            )

    @staticmethod
    def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
        return value.exp().sub(1.0).log()

    @property
    def num_examples(self) -> int:
        return self.config.dpc * self.num_labels

    @property
    def attention_loss_lambda(self) -> float:
        return self.config.attention_loss_lambda

    def to(self, device: torch.device | str):
        device = torch.device(device)
        for name in ("input_embeds", "label_ids", "label_logits", "lr_logits", "attention_logits"):
            value = getattr(self, name)
            if value is not None:
                if value.device == device:
                    continue
                grad = value.grad
                requires_grad = value.requires_grad
                value = value.detach().to(device)
                if value.is_floating_point():
                    value = value.requires_grad_(requires_grad)
                    value.grad = grad.to(device) if grad is not None else None
                setattr(self, name, value)
        return self

    def parameters(self) -> list[torch.Tensor]:
        params = [self.input_embeds, self.lr_logits]
        if self.label_logits is not None:
            params.append(self.label_logits)
        if self.attention_logits is not None:
            params.append(self.attention_logits)
        return params

    def parameter_groups(
        self,
        *,
        lr_input_embeds: float,
        lr_labels: float,
        lr_lrs: float,
        lr_attention_labels: float,
        weight_decay: float,
    ) -> list[dict]:
        groups = [
            {
                "params": [self.input_embeds],
                "lr": lr_input_embeds,
                "weight_decay": weight_decay,
            },
            {"params": [self.lr_logits], "lr": lr_lrs, "weight_decay": 0.0},
        ]
        if self.label_logits is not None:
            groups.append({"params": [self.label_logits], "lr": lr_labels, "weight_decay": 0.0})
        if self.attention_logits is not None:
            groups.append(
                {
                    "params": [self.attention_logits],
                    "lr": lr_attention_labels,
                    "weight_decay": weight_decay,
                }
            )
        return groups

    def initialize_embeddings(self, embeddings: torch.Tensor):
        if embeddings.shape != self.input_embeds.shape:
            raise ValueError(f"Expected embeddings with shape {self.input_embeds.shape}, got {embeddings.shape}.")
        with torch.no_grad():
            self.input_embeds.copy_(embeddings.to(self.input_embeds.device))

    def get_batch(self, step: int) -> dict[str, torch.Tensor | None]:
        indices = self.get_batch_indices(step).to(self.input_embeds.device)
        batch = {
            "inputs_embeds": self.input_embeds[indices],
            "labels": self._labels(indices),
            "attention_labels": self._attention_labels(indices),
            "lr": self._lr(step),
        }
        return batch

    def get_batch_indices(self, step: int) -> torch.Tensor:
        batch_size = self.num_labels * self.train_config.batch_size_per_label
        if self.config.fix_order:
            cycle = step % (self.num_examples // batch_size)
            return torch.arange(batch_size * cycle, batch_size * (cycle + 1))
        return torch.randperm(self.num_examples)[:batch_size]

    def _labels(self, indices: torch.Tensor) -> torch.Tensor:
        if self.label_logits is None:
            return self.label_ids.to(indices.device)[indices]
        return self.label_logits[indices].softmax(dim=-1)

    def _attention_labels(self, indices: torch.Tensor) -> torch.Tensor | None:
        if self.attention_logits is None:
            return None
        return self.attention_logits[indices].softmax(dim=-1)

    def _lr(self, step: int) -> torch.Tensor:
        if self.config.learn_inner_lrs:
            return F.softplus(self.lr_logits[step])

        scale = 1.0
        if self.config.lr_linear_decay:
            scale -= step / self.train_config.inner_steps
        return F.softplus(self.lr_logits) * scale

    def data_dict(self, detach: bool = False) -> dict[str, torch.Tensor]:
        data = {
            "synthetic_embeddings": self.input_embeds,
            "synthetic_labels": self.label_ids,
            "inner_lrs": F.softplus(self.lr_logits),
        }
        if self.label_logits is not None:
            data["synthetic_label_logits"] = self.label_logits
        if self.attention_logits is not None:
            data["synthetic_attention_logits"] = self.attention_logits
        if detach:
            return {k: v.detach().cpu() for k, v in data.items()}
        return data

    def save_pretrained(self, save_path: str | os.PathLike):
        os.makedirs(save_path, exist_ok=True)
        config = {
            "config": asdict(self.config),
            "train_config": asdict(self.train_config),
            "num_labels": self.num_labels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
        }
        with open(os.path.join(save_path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        torch.save(self.data_dict(detach=True), os.path.join(save_path, "data_dict.pt"))

    def load_data_dict(self, data: dict[str, torch.Tensor]):
        with torch.no_grad():
            self.input_embeds.copy_(data["synthetic_embeddings"])
            self.label_ids.copy_(data["synthetic_labels"])
            self.lr_logits.copy_(self._inverse_softplus(data["inner_lrs"]))
            if self.label_logits is not None and "synthetic_label_logits" in data:
                self.label_logits.copy_(data["synthetic_label_logits"])
            if self.attention_logits is not None and "synthetic_attention_logits" in data:
                self.attention_logits.copy_(data["synthetic_attention_logits"])

    @classmethod
    def from_pretrained(cls, save_path: str | os.PathLike):
        with open(os.path.join(save_path, "config.json")) as f:
            config = json.load(f)
        obj = cls(
            config=EmbeddingDistillationDataConfig(**config["config"]),
            train_config=EmbeddingLearnerTrainConfig(**config["train_config"]),
            num_labels=config["num_labels"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
        )
        obj.load_data_dict(torch.load(os.path.join(save_path, "data_dict.pt"), map_location="cpu"))
        return obj
