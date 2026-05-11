from datasets import Dataset

from text_distillation.data.datasets import get_dataset_info, list_supported_datasets, make_tiny_subset


def make_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "text": [f"text {index}" for index in range(12)],
            "label": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )


def test_make_tiny_subset_stratified_size_and_labels():
    dataset = make_dataset()

    subset = make_tiny_subset(dataset, n_per_class=2, seed=123)

    assert len(subset) == 8
    assert sorted(subset["label"]) == [0, 0, 1, 1, 2, 2, 3, 3]


def test_make_tiny_subset_is_deterministic():
    dataset = make_dataset()

    first = make_tiny_subset(dataset, total_size=5, seed=123)
    second = make_tiny_subset(dataset, total_size=5, seed=123)

    assert first["text"] == second["text"]
    assert first["label"] == second["label"]


def test_supported_dataset_info_contains_paper_datasets():
    assert {"sst2", "qqp", "mnli-m"}.issubset(set(list_supported_datasets()))

    assert get_dataset_info("ag-news").name == "ag_news"
    assert get_dataset_info("ag_news").name == "ag_news"

    qqp_info = get_dataset_info("qqp")
    mnli_info = get_dataset_info("mnli-m")

    assert qqp_info.hf_path == "glue"
    assert qqp_info.hf_config == "qqp"
    assert qqp_info.text_columns == ("question1", "question2")
    assert qqp_info.metric_name == "accuracy_f1_average"

    assert mnli_info.eval_split == "validation_matched"
    assert mnli_info.text_columns == ("premise", "hypothesis")
