"""Registry of model profiles used in baseline experiments.

A `ModelProfile` describes per-model metadata that the rest of the codebase
needs to make consistent choices: which token to use for embedding pooling,
sensible batch sizes for a T4 16GB starting point, and whether fp16 is safe.

The registry is stateless: profiles are pure metadata, the underlying weights
are loaded on demand via `AutoModel.from_pretrained(model_name)` elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PoolingStrategy = Literal["first_token", "last_token", "mean"]
ModelFamily = Literal["bert", "roberta", "albert", "deberta-v3", "xlnet"]


@dataclass(frozen=True)
class ModelProfile:
    model_name: str
    family: ModelFamily
    embedding_pooling: PoolingStrategy
    recommended_max_length: int = 128
    recommended_train_batch_size: int = 64
    recommended_embedding_batch_size: int = 128
    supports_fp16: bool = True


_DEFAULT_PROFILES: list[ModelProfile] = [
    ModelProfile(
        model_name="bert-base-uncased",
        family="bert",
        embedding_pooling="first_token",
    ),
    ModelProfile(
        model_name="bert-large-uncased",
        family="bert",
        embedding_pooling="first_token",
        recommended_train_batch_size=16,
        recommended_embedding_batch_size=32,
    ),
    ModelProfile(
        model_name="roberta-base",
        family="roberta",
        embedding_pooling="first_token",
    ),
    ModelProfile(
        model_name="albert-base-v2",
        family="albert",
        embedding_pooling="first_token",
    ),
    ModelProfile(
        model_name="microsoft/deberta-v3-base",
        family="deberta-v3",
        embedding_pooling="first_token",
    ),
    ModelProfile(
        model_name="xlnet-base-cased",
        family="xlnet",
        embedding_pooling="last_token",
    ),
]


MODEL_REGISTRY: dict[str, ModelProfile] = {p.model_name: p for p in _DEFAULT_PROFILES}


def register_model(profile: ModelProfile, *, overwrite: bool = False) -> None:
    """Add or replace a profile in the registry."""
    if profile.model_name in MODEL_REGISTRY and not overwrite:
        raise ValueError(
            f"model {profile.model_name!r} is already registered; pass overwrite=True to replace it"
        )
    MODEL_REGISTRY[profile.model_name] = profile


def get_model_profile(model_name: str) -> ModelProfile:
    """Return the registered profile, or a sensible BERT-like default."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    return ModelProfile(
        model_name=model_name,
        family="bert",
        embedding_pooling="first_token",
    )


def list_models() -> list[str]:
    return list(MODEL_REGISTRY.keys())
