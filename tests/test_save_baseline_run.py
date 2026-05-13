"""Tests for `save_baseline_run` — only persistence, no dataset downloads."""

from dataclasses import replace

from datasets import Dataset, load_from_disk

from text_distillation.data.datasets import TextClassificationDatasetInfo
from text_distillation.experiments import BaselineData, save_baseline_run
from text_distillation.saving import create_run_dir
from text_distillation.timing import TimingTracker
from text_distillation.utils import load_json


DATASET_INFO = TextClassificationDatasetInfo(
    name="toy",
    hf_path="local/toy",
    hf_config=None,
    train_split="train",
    eval_split="test",
    test_split="test",
    text_columns=("text",),
    label_column="label",
    metric_name="accuracy",
)


def _make_data() -> BaselineData:
    train_pool = Dataset.from_dict({"text": ["a", "b", "c", "d"], "label": [0, 1, 0, 1]})
    eval_dataset = Dataset.from_dict({"text": ["x", "y"], "label": [0, 1]})
    return BaselineData(
        dataset_info=DATASET_INFO,
        train_pool=train_pool,
        eval_dataset=eval_dataset,
        label_names=["neg", "pos"],
        num_labels=2,
        full_train_size=4,
        train_pool_size=4,
    )


def test_save_baseline_run_writes_config_metrics_and_distilled(tmp_path):
    data = _make_data()
    distilled = Dataset.from_dict({"text": ["a", "c"], "label": [0, 0]})
    run_dir = create_run_dir(tmp_path, "stratified_random_toy_bert_base_uncased_k1")

    tracker = TimingTracker()
    tracker.add("selection_sec", 0.5)
    tracker.add("training_sec", 1.5)

    row = save_baseline_run(
        run_dir=run_dir,
        data=data,
        method_name="stratified_random",
        model_name="bert-base-uncased",
        k_per_class=1,
        seed=42,
        train_dataset=distilled,
        metrics={"accuracy": 0.5, "f1_macro": 0.5, "train_loss": 0.1},
        tracker=tracker,
    )

    config = load_json(run_dir / "config.json")
    metrics = load_json(run_dir / "metrics.json")

    assert config["method_name"] == "stratified_random"
    assert config["model_name"] == "bert-base-uncased"
    assert config["model_family"] == "bert"
    assert config["dataset_name"] == "toy"
    assert config["k_total"] == 2
    assert config["full_train_size"] == 4
    assert config["compression_ratio"] == 2.0
    assert config["dpc"] == 1

    assert metrics["accuracy"] == 0.5
    assert metrics["timings"] == {"selection_sec": 0.5, "training_sec": 1.5}

    assert row["accuracy"] == 0.5
    assert row["compression_ratio"] == 2.0

    restored = load_from_disk(str(run_dir / "distilled_dataset"))
    assert restored["text"] == ["a", "c"]


def test_save_baseline_run_skips_distilled_for_full_data(tmp_path):
    data = _make_data()
    run_dir = create_run_dir(tmp_path, "full_data_toy_bert_base_uncased_full")
    save_baseline_run(
        run_dir=run_dir,
        data=data,
        method_name="full_data",
        model_name="bert-base-uncased",
        k_per_class=None,
        seed=42,
        train_dataset=data.train_pool,
        metrics={"accuracy": 0.9, "f1_macro": 0.9, "train_loss": 0.05},
    )
    assert not (run_dir / "distilled_dataset").exists(), "full_data must not persist a separate subset"
    config = load_json(run_dir / "config.json")
    assert config["dpc"] is None
    assert config["compression_ratio"] == 1.0


def test_save_baseline_run_accepts_extra_config(tmp_path):
    data = _make_data()
    run_dir = create_run_dir(tmp_path, "kcenter_cls_toy_bert_base_uncased_k1")
    save_baseline_run(
        run_dir=run_dir,
        data=data,
        method_name="kcenter_cls",
        model_name="bert-base-uncased",
        k_per_class=1,
        seed=42,
        train_dataset=Dataset.from_dict({"text": ["a"], "label": [0]}),
        metrics={"accuracy": 0.0, "f1_macro": 0.0, "train_loss": 0.0},
        extra_config={"embedding_model_name": "bert-base-uncased"},
    )
    config = load_json(run_dir / "config.json")
    assert config["embedding_model_name"] == "bert-base-uncased"


def test_baseline_data_train_pool_size_can_differ_from_full(tmp_path):
    """When a smoke-check subsamples the pool, both sizes are tracked."""
    base = _make_data()
    sub = replace(base, train_pool_size=2)
    run_dir = create_run_dir(tmp_path, "stratified_random_toy_bert_base_uncased_k1_smoke")
    save_baseline_run(
        run_dir=run_dir,
        data=sub,
        method_name="stratified_random",
        model_name="bert-base-uncased",
        k_per_class=1,
        seed=42,
        train_dataset=Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]}),
        metrics={"accuracy": 1.0, "f1_macro": 1.0, "train_loss": 0.0},
    )
    config = load_json(run_dir / "config.json")
    assert config["full_train_size"] == 4
    assert config["train_pool_size"] == 2
    assert config["train_size"] == 2
