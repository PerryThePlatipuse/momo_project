from pathlib import Path

from datasets import Dataset

from text_distillation.dilm.official import (
    load_official_dilm_dataset,
    load_paper_vanilla_lm_dataset,
    official_dilm_paths,
    paper_vanilla_lm_paths,
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


def test_paper_vanilla_lm_paths_are_local_to_vanilla_root(tmp_path):
    run = paper_vanilla_lm_paths(
        "sst2",
        dpc=10,
        seed=3,
        output_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
    )

    assert run.task_name == "sst2"
    assert "dilm.lm" in str(run.lm_save_dir)
    assert run.dataset_dir == (
        tmp_path
        / "artifacts"
        / "test.bert-base-uncased.sst2"
        / "dilm.lm"
        / "dpc_10_seed_3_paper"
        / "dataset"
    )


def test_load_paper_vanilla_lm_dataset_renames_labels(tmp_path):
    run = paper_vanilla_lm_paths(
        "sst2",
        dpc=5,
        seed=42,
        output_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
    )
    run.dataset_dir.mkdir(parents=True)
    Dataset.from_dict({"sentence": ["bad"], "labels": [0]}).to_json(
        str(run.dataset_dir / "dataset_0.json")
    )

    dataset = load_paper_vanilla_lm_dataset(
        "sst2",
        dpc=5,
        seed=42,
        output_root=tmp_path / "artifacts",
        data_root=tmp_path / "data",
    )

    assert dataset.column_names == ["sentence", "label"]
    assert dataset["label"] == [0]
