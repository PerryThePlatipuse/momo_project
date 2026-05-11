from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise_distances
from tqdm.auto import tqdm

from text_distillation.data.dataloaders import create_text_dataloader
from text_distillation.data.transforms import TextColumns, join_text_columns, normalize_text_columns
from text_distillation.utils import get_device, move_batch_to_device, set_seed


def select_random(dataset: Any, k: int, seed: int = 42):
    """Select `k` random examples from a dataset."""
    if k > len(dataset):
        raise ValueError(f"k={k} exceeds dataset size {len(dataset)}")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=k, replace=False).tolist()
    return dataset.select(indices)


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
):
    """Select k-center examples per class over encoder `[CLS]` embeddings."""
    embeddings = compute_cls_embeddings(
        dataset=dataset,
        text_column=text_column,
        text_columns=text_columns,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        normalize=normalize,
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


def compute_cls_embeddings(
    dataset: Any,
    text_column: str = "text",
    text_columns: TextColumns | None = None,
    model_name: str = "bert-base-uncased",
    batch_size: int = 64,
    max_length: int = 128,
    device: str | None = None,
    normalize: bool = True,
) -> np.ndarray:
    """Compute `[CLS]` embeddings for every text in a dataset."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch
    from transformers import AutoModel, AutoTokenizer

    device = device or get_device()
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

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Computing CLS embeddings"):
            batch = move_batch_to_device(batch, device)
            if use_cuda_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs = model(**batch)
            else:
                outputs = model(**batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().float().numpy()
            all_embeddings.append(cls_embeddings)

    embeddings = np.concatenate(all_embeddings, axis=0)
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
    return embeddings


def _indices_by_label(dataset: Any, label_column: str) -> dict[Any, list[int]]:
    indices_by_label: dict[Any, list[int]] = defaultdict(list)
    for index, label in enumerate(dataset[label_column]):
        indices_by_label[label].append(index)
    return dict(indices_by_label)


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
