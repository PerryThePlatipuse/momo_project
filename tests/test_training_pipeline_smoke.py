import os

import pytest

from text_distillation.data import load_ag_news, make_tiny_subset
from text_distillation.data.datasets import get_label_names
from text_distillation.distillation import (
    select_kcenter_embeddings,
    select_kcenter_tfidf,
    select_random,
    select_stratified_random,
)
from text_distillation.model import train_text_classifier


pytestmark = pytest.mark.integration


def _require_training_smoke_enabled():
    if os.environ.get("RUN_TRAINING_SMOKE") != "1":
        pytest.skip("Set RUN_TRAINING_SMOKE=1 to run end-to-end training smoke tests.")


def _assert_training_metrics(metrics: dict[str, float]) -> None:
    assert "train_loss" in metrics
    assert "accuracy" in metrics
    assert "f1_macro" in metrics
    assert metrics["train_loss"] >= 0.0
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_macro"] <= 1.0


def test_all_baselines_train_on_100_examples(tmp_path):
    _require_training_smoke_enabled()

    dataset = load_ag_news()
    train_pool = make_tiny_subset(dataset["train"], n_per_class=40, seed=42)
    full_tiny_train = make_tiny_subset(dataset["train"], n_per_class=25, seed=43)
    eval_dataset = make_tiny_subset(dataset["test"], n_per_class=5, seed=44)
    label_names = get_label_names(dataset["train"])

    model_name = "hf-internal-testing/tiny-random-bert"
    k_per_class = 25
    k_total = k_per_class * len(label_names)

    baseline_train_sets = {
        "full_tiny": full_tiny_train,
        "random": select_random(train_pool, k=k_total, seed=45),
        "stratified_random": select_stratified_random(
            train_pool,
            k_per_class=k_per_class,
            seed=46,
        ),
        "kcenter_tfidf": select_kcenter_tfidf(
            train_pool,
            k_per_class=k_per_class,
            seed=47,
            max_features=1_000,
        ),
        "kcenter_cls": select_kcenter_embeddings(
            train_pool,
            k_per_class=k_per_class,
            model_name=model_name,
            batch_size=16,
            max_length=32,
            seed=48,
        ),
    }

    for baseline_name, train_dataset in baseline_train_sets.items():
        assert len(train_dataset) == 100

        _, metrics = train_text_classifier(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_name=model_name,
            output_dir=tmp_path / baseline_name,
            num_labels=len(label_names),
            label_names=label_names,
            max_length=32,
            num_train_epochs=1.0,
            train_batch_size=16,
            eval_batch_size=16,
            seed=100,
            save_model=False,
        )
        _assert_training_metrics(metrics)

