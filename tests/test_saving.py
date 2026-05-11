from datasets import Dataset, load_from_disk

from text_distillation.saving import (
    create_run_dir,
    save_distilled_dataset,
    save_experiment_config,
    save_metrics,
)
from text_distillation.utils import load_json


def test_save_run_artifacts(tmp_path):
    run_dir = create_run_dir(tmp_path, "toy_run")
    dataset = Dataset.from_dict({"text": ["a", "b"], "label": [0, 1]})

    config_path = save_experiment_config({"seed": 42, "method": "toy"}, run_dir)
    metrics_path = save_metrics({"accuracy": 0.5, "f1_macro": 0.3333}, run_dir)
    dataset_path = save_distilled_dataset(dataset, run_dir)

    assert load_json(config_path) == {"seed": 42, "method": "toy"}
    assert load_json(metrics_path) == {"accuracy": 0.5, "f1_macro": 0.3333}

    restored = load_from_disk(str(dataset_path))
    assert restored["text"] == ["a", "b"]
    assert restored["label"] == [0, 1]

