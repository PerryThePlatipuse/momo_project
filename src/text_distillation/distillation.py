from __future__ import annotations

import os
from collections import defaultdict
from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

from text_distillation.data.dataloaders import create_text_dataloader
from text_distillation.data.transforms import TextColumns, join_text_columns, normalize_text_columns
from text_distillation.model.registry import PoolingStrategy, get_model_profile
from text_distillation.timing import TimingTracker
from text_distillation.utils import get_device, move_batch_to_device, set_seed


SELECTION_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_selection(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a selection function under `name` for use by `run_baseline_experiment`.

    The registered callable must accept ``dataset`` as the first positional
    argument and ``seed`` as a keyword. All other arguments are method-specific
    and routed through `ExperimentConfig.method_kwargs`.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in SELECTION_REGISTRY:
            raise ValueError(f"selection method {name!r} is already registered")
        SELECTION_REGISTRY[name] = fn
        return fn

    return decorator


def get_selection_fn(name: str) -> Callable[..., Any]:
    if name not in SELECTION_REGISTRY:
        known = ", ".join(sorted(SELECTION_REGISTRY)) or "<none>"
        raise KeyError(f"unknown selection method {name!r}; known: {known}")
    return SELECTION_REGISTRY[name]


def list_selection_methods() -> list[str]:
    return sorted(SELECTION_REGISTRY)


@register_selection("random")
def select_random(dataset: Any, k: int, seed: int = 42):
    """Select `k` random examples from a dataset."""
    if k > len(dataset):
        raise ValueError(f"k={k} exceeds dataset size {len(dataset)}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=k, replace=False).tolist()
    return dataset.select(indices)


@register_selection("stratified_random")
def select_stratified_random(
    dataset: Any,
    label_column: str = "label",
    k_per_class: int = 10,
    seed: int = 42,
):
    """Select `k_per_class` random examples from every class."""
    indices_by_label = _indices_by_label(dataset, label_column)
    rng = np.random.default_rng(seed)

    selected: list[int] = []
    for label in sorted(indices_by_label):
        label_indices = indices_by_label[label]
        if k_per_class > len(label_indices):
            raise ValueError(
                f"k_per_class={k_per_class} exceeds class {label} size {len(label_indices)}"
            )
        selected.extend(rng.choice(label_indices, size=k_per_class, replace=False).tolist())

    return dataset.select(selected)


@register_selection("kcenter_tfidf")
def select_kcenter_tfidf(
    dataset: Any,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    label_column: str = "label",
    k_per_class: int = 10,
    seed: int = 42,
    max_features: int = 50_000,
):
    """Select k-center examples per class using TF-IDF vectors."""
    columns = normalize_text_columns(text_column=text_column, text_columns=text_columns)
    texts = join_text_columns(dataset, columns)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    features = vectorizer.fit_transform(texts)

    selected: list[int] = []
    for label, class_indices in sorted(_indices_by_label(dataset, label_column).items()):
        if k_per_class > len(class_indices):
            raise ValueError(
                f"k_per_class={k_per_class} exceeds class {label} size {len(class_indices)}"
            )
        class_features = features[class_indices]
        local_selected = _greedy_k_center(class_features, k=k_per_class, seed=seed)
        selected.extend(class_indices[index] for index in local_selected)

    return dataset.select(selected)


@register_selection("herding")
def select_herding(
    dataset: Any,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    label_column: str = "label",
    k_per_class: int = 10,
    model_name: str = "bert-base-uncased",
    batch_size: int = 64,
    max_length: int = 128,
    seed: int = 42,
    device: str | None = None,
    normalize: bool = True,
    pooling: PoolingStrategy | None = None,
    tracker: TimingTracker | None = None,
):
    """Select examples per class via herding (Welling 2009) over encoder embeddings.

    Greedy choice: at step ``t`` pick the example whose addition keeps the running
    mean of selected embeddings closest to the class mean. With L2-normalized
    embeddings this matches Welling's ``argmax_i <x_i, w_t>`` update where
    ``w_t = w_{t-1} + mu - x_{i*}``.
    """
    embeddings = compute_text_embeddings(
        dataset=dataset,
        text_column=text_column,
        text_columns=text_columns,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        normalize=normalize,
        pooling=pooling,
        tracker=tracker,
    )

    selected: list[int] = []
    for label, class_indices in sorted(_indices_by_label(dataset, label_column).items()):
        if k_per_class > len(class_indices):
            raise ValueError(
                f"k_per_class={k_per_class} exceeds class {label} size {len(class_indices)}"
            )
        class_embeddings = embeddings[class_indices]
        local_selected = _greedy_herding(class_embeddings, k=k_per_class)
        selected.extend(class_indices[index] for index in local_selected)

    return dataset.select(selected)


@register_selection("kcenter_cls")
def select_kcenter_embeddings(
    dataset: Any,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    label_column: str = "label",
    k_per_class: int = 10,
    model_name: str = "bert-base-uncased",
    batch_size: int = 64,
    max_length: int = 128,
    seed: int = 42,
    device: str | None = None,
    normalize: bool = True,
    pooling: PoolingStrategy | None = None,
    tracker: TimingTracker | None = None,
):
    """Select k-center examples per class over encoder text embeddings.

    `pooling` defaults to the strategy registered for `model_name`:
    `"first_token"` for BERT/RoBERTa/ALBERT/DeBERTa and `"last_token"` for XLNet.
    """
    embeddings = compute_text_embeddings(
        dataset=dataset,
        text_column=text_column,
        text_columns=text_columns,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        normalize=normalize,
        pooling=pooling,
        tracker=tracker,
    )

    selected: list[int] = []
    for label, class_indices in sorted(_indices_by_label(dataset, label_column).items()):
        if k_per_class > len(class_indices):
            raise ValueError(
                f"k_per_class={k_per_class} exceeds class {label} size {len(class_indices)}"
            )
        class_embeddings = embeddings[class_indices]
        local_selected = _greedy_k_center(class_embeddings, k=k_per_class, seed=seed)
        selected.extend(class_indices[index] for index in local_selected)

    return dataset.select(selected)


def compute_text_embeddings(
    dataset: Any,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    model_name: str = "bert-base-uncased",
    batch_size: int = 64,
    max_length: int = 128,
    device: str | None = None,
    normalize: bool = True,
    pooling: PoolingStrategy | None = None,
    tracker: TimingTracker | None = None,
) -> np.ndarray:
    """Compute pooled text embeddings for every example in `dataset`.

    `pooling` controls how token states are reduced to one vector:
    - `"first_token"`: hidden state at position 0 (CLS for BERT-family).
    - `"last_token"`: hidden state at the last *non-padding* position (XLNet-style).
    - `"mean"`: attention-masked mean over tokens.
    Defaults to the strategy from the model registry.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModel, AutoTokenizer

    if pooling is None:
        pooling = get_model_profile(model_name).embedding_pooling

    device = device or get_device()
    with _maybe_measure(tracker, "embedding_load_sec"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.to(device)
        model.eval()
    columns = normalize_text_columns(text_column=text_column, text_columns=text_columns)

    def tokenize_batch(batch):
        if len(columns) == 1:
            return tokenizer(
                batch[columns[0]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        if len(columns) == 2:
            return tokenizer(
                batch[columns[0]],
                batch[columns[1]],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
        texts = join_text_columns(batch, columns)
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    with _maybe_measure(tracker, "embedding_tokenize_sec"):
        tokenized = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing for embeddings",
        )
        tokenized.set_format(type="torch")

    dataloader = create_text_dataloader(
        tokenized,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=False,
    )

    all_embeddings: list[np.ndarray] = []
    use_cuda_amp = device == "cuda"

    with _maybe_measure(tracker, "embedding_forward_sec"):
        with torch.inference_mode():
            for batch in tqdm(dataloader, desc="Computing text embeddings"):
                batch = move_batch_to_device(batch, device)
                if use_cuda_amp:
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        outputs = model(**batch)
                else:
                    outputs = model(**batch)
                pooled = _pool_hidden_states(
                    outputs.last_hidden_state,
                    attention_mask=batch.get("attention_mask"),
                    pooling=pooling,
                )
                all_embeddings.append(pooled.detach().cpu().float().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings


def _pool_hidden_states(
    last_hidden_state: Any,
    attention_mask: Any,
    pooling: PoolingStrategy,
) -> Any:
    import torch

    if pooling == "first_token":
        return last_hidden_state[:, 0, :]
    if pooling == "last_token":
        if attention_mask is None:
            return last_hidden_state[:, -1, :]
        lengths = attention_mask.sum(dim=1) - 1
        lengths = lengths.clamp(min=0)
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        return last_hidden_state[batch_indices, lengths, :]
    if pooling == "mean":
        if attention_mask is None:
            return last_hidden_state.mean(dim=1)
        mask = attention_mask.to(last_hidden_state.dtype).unsqueeze(-1)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
    raise ValueError(f"unknown pooling strategy {pooling!r}")


def _maybe_measure(tracker: TimingTracker | None, name: str):
    return tracker.measure(name) if tracker is not None else nullcontext()


def _indices_by_label(dataset: Any, label_column: str) -> dict[Any, list[int]]:
    indices_by_label: dict[Any, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset[label_column]):
        indices_by_label[label].append(index)
    return dict(indices_by_label)


def _greedy_herding(features: np.ndarray, k: int) -> list[int]:
    """Greedy herding (Welling 2009) over dense feature vectors.

    Deterministic: no random initialization — the choice is fully determined
    by the feature matrix and ``k``. Designed for L2-normalized embeddings;
    works on raw embeddings too but the ``argmax dot product`` step is most
    meaningful when norms are comparable across points.
    """
    n_samples = features.shape[0]
    if k > n_samples:
        raise ValueError(f"k={k} exceeds number of samples {n_samples}")
    if k == n_samples:
        return list(range(n_samples))

    features = np.asarray(features)
    mean = features.mean(axis=0)
    residual = mean.copy()
    selected: list[int] = []
    selected_mask = np.zeros(n_samples, dtype=bool)
    for _ in range(k):
        scores = features @ residual
        scores[selected_mask] = -np.inf
        next_index = int(np.argmax(scores))
        selected.append(next_index)
        selected_mask[next_index] = True
        residual = residual + mean - features[next_index]
    return selected


def _greedy_k_center(features: Any, k: int, seed: int = 42) -> list[int]:
    """Greedy farthest-first k-center selection over dense or sparse features."""
    n_samples = features.shape[0]
    if k > n_samples:
        raise ValueError(f"k={k} exceeds number of samples {n_samples}")
    if k == n_samples:
        return list(range(n_samples))

    set_seed(seed)
    rng = np.random.default_rng(seed)
    first_index = int(rng.integers(n_samples))
    selected = [first_index]

    min_distances = pairwise_distances(features, features[[first_index]], metric="euclidean").reshape(-1)
    min_distances[first_index] = 0.0

    for _ in range(1, k):
        next_index = int(np.argmax(min_distances))
        selected.append(next_index)
        distances = pairwise_distances(features, features[[next_index]], metric="euclidean").reshape(-1)
        min_distances = np.minimum(min_distances, distances)
        min_distances[selected] = 0.0

    return selected
