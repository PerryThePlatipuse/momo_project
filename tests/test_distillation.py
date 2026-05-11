from datasets import Dataset

from text_distillation.distillation import (
    select_kcenter_tfidf,
    select_random,
    select_stratified_random,
)


def make_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "text": [
                "sports team wins championship",
                "basketball player scores points",
                "football match goes overtime",
                "new processor chip announced",
                "software update fixes bug",
                "startup launches cloud platform",
                "central bank changes rates",
                "stocks close higher today",
                "market analysts expect growth",
            ],
            "label": [0, 0, 0, 1, 1, 1, 2, 2, 2],
        }
    )


def test_select_random_size_and_determinism():
    dataset = make_dataset()

    first = select_random(dataset, k=4, seed=7)
    second = select_random(dataset, k=4, seed=7)

    assert len(first) == 4
    assert first["text"] == second["text"]


def test_select_stratified_random_balances_classes():
    dataset = make_dataset()

    subset = select_stratified_random(dataset, k_per_class=2, seed=7)

    assert len(subset) == 6
    assert sorted(subset["label"]) == [0, 0, 1, 1, 2, 2]


def test_select_kcenter_tfidf_balances_classes():
    dataset = make_dataset()

    subset = select_kcenter_tfidf(dataset, k_per_class=1, seed=7)

    assert len(subset) == 3
    assert sorted(subset["label"]) == [0, 1, 2]

