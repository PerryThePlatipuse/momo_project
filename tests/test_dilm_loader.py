"""Tests for the DiLM pre-generated dataset loader.

These tests use a fake `dilm_data_root` populated with a JSONL that mimics the
real DiLM-main/DiLM-synthetic-data/ layout, so they don't depend on the actual
shipped files being present.
"""

import json

import pytest

from text_distillation.dilm import distill_dilm


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _fake_root(tmp_path):
    # Match the layout: <task>/dilm.dc/dpc_{K}.<long_run_id>/dataset/dataset_<i>.json
    base = tmp_path / "DiLM-main" / "DiLM-synthetic-data"
    for task, records in [
        ("sst2", [
            {"sentence": "fantastic film", "labels": 1},
            {"sentence": "boring as hell", "labels": 0},
            {"sentence": "lovely cast", "labels": 1},
            {"sentence": "wasted my time", "labels": 0},
        ]),
        ("qqp", [
            {"question1": "How to learn Python?", "question2": "Best Python tutorial?", "labels": 1},
            {"question1": "Why is the sky blue?", "question2": "Recipe for pancakes?", "labels": 0},
        ]),
        ("mnli", [
            {"premise": "He left early.", "hypothesis": "He did not stay.", "labels": 0},
            {"premise": "She is reading.", "hypothesis": "She is asleep.", "labels": 2},
            {"premise": "It rained.", "hypothesis": "Water fell.", "labels": 0},
        ]),
    ]:
        run_dir = base / task / "dilm.dc" / "dpc_5.fake_run_id"
        _write_jsonl(run_dir / "dataset" / "dataset_0.json", records)
    return base


def test_dilm_loads_sst2_into_dataset(tmp_path):
    root = _fake_root(tmp_path)
    ds = distill_dilm(
        dataset=None,
        dataset_name="sst2",
        k_per_class=5,
        dataset_index=0,
        dilm_data_root=root,
    )
    assert len(ds) == 4
    assert set(ds.column_names) == {"sentence", "label"}
    assert sorted(ds["label"]) == [0, 0, 1, 1]


def test_dilm_renames_labels_field_to_project_label_column(tmp_path):
    root = _fake_root(tmp_path)
    # Project uses `label_column="label"` everywhere; DiLM JSONL uses `labels`.
    ds = distill_dilm(
        dataset=None, dataset_name="sst2", k_per_class=5, dataset_index=0,
        dilm_data_root=root,
    )
    assert "labels" not in ds.column_names
    assert "label" in ds.column_names


def test_dilm_keeps_pair_columns_for_qqp_and_mnli(tmp_path):
    root = _fake_root(tmp_path)
    qqp = distill_dilm(dataset=None, dataset_name="qqp", k_per_class=5, dataset_index=0,
                      dilm_data_root=root)
    assert {"question1", "question2", "label"}.issubset(set(qqp.column_names))

    mnli = distill_dilm(dataset=None, dataset_name="mnli-m", k_per_class=5, dataset_index=0,
                       dilm_data_root=root)
    assert {"premise", "hypothesis", "label"}.issubset(set(mnli.column_names))


def test_dilm_rejects_unsupported_dpc(tmp_path):
    root = _fake_root(tmp_path)
    with pytest.raises(ValueError, match=r"DPC in \{5, 10, 20\}"):
        distill_dilm(dataset=None, dataset_name="sst2", k_per_class=15,
                    dataset_index=0, dilm_data_root=root)


def test_dilm_rejects_unsupported_dataset(tmp_path):
    root = _fake_root(tmp_path)
    with pytest.raises(ValueError, match="ag_news"):
        distill_dilm(dataset=None, dataset_name="ag_news", k_per_class=50,
                    dataset_index=0, dilm_data_root=root)


def test_dilm_rejects_out_of_range_dataset_index(tmp_path):
    root = _fake_root(tmp_path)
    with pytest.raises(ValueError, match="0..19"):
        distill_dilm(dataset=None, dataset_name="sst2", k_per_class=5,
                    dataset_index=42, dilm_data_root=root)
