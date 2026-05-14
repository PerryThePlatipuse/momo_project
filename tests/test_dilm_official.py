from pathlib import Path

from datasets import Dataset

from text_distillation.dilm.official import (
    load_official_dilm_dataset,
    official_dilm_paths,
)


def test_official_dilm_paths_map_mnli_m_to_official_mnli(tmp_path):
    run = official_dilm_paths(
        "mnli-m",
        dpc=20,
        seed=7,
        output_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
    )

    assert run.task_name == "mnli"
    assert run.lm_generator_dir == (
        tmp_path
        / "artifacts"
        / "train.gpt2.bert-base-uncased.mnli"
        / "dilm.lm"
        / "step_80000_seed_7"
        / "generator"
    )
    assert run.dataset_dir.name == "dataset"


def test_load_official_dilm_dataset_renames_labels(tmp_path):
    run = official_dilm_paths(
        "sst2",
        dpc=5,
        seed=42,
        output_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
    )
    run.dataset_dir.mkdir(parents=True)
    Dataset.from_dict({"sentence": ["great"], "labels": [1]}).to_json(
        str(run.dataset_dir / "dataset_0.json")
    )

    dataset = load_official_dilm_dataset(
        "sst2",
        dpc=5,
        seed=42,
        output_root=Path(tmp_path / "artifacts"),
        data_root=Path(tmp_path / "data"),
    )

    assert dataset.column_names == ["sentence", "label"]
    assert dataset["label"] == [1]
