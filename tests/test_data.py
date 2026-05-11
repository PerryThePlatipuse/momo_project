from datasets import Dataset

from text_distillation.data.datasets import make_tiny_subset


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

