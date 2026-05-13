import numpy as np
from datasets import Dataset

from text_distillation.distillation import (
    _greedy_herding,
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


def test_greedy_herding_picks_close_to_class_mean():
    rng = np.random.default_rng(0)
    # 100 points around the origin plus one outlier far away. Class mean is
    # close to the origin, so herding must avoid the outlier when k is small.
    cluster = rng.normal(scale=0.1, size=(100, 4))
    features = np.vstack([cluster, np.array([[10.0, 10.0, 10.0, 10.0]])])

    selected = _greedy_herding(features, k=5)
    assert 100 not in selected, "herding picked the obvious outlier"
    assert len(selected) == len(set(selected)) == 5


def test_greedy_herding_is_deterministic():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(20, 6))
    first = _greedy_herding(features, k=7)
    second = _greedy_herding(features, k=7)
    assert first == second


def test_greedy_herding_returns_full_set_when_k_equals_n():
    features = np.eye(4)
    assert _greedy_herding(features, k=4) == [0, 1, 2, 3]


def test_select_kcenter_tfidf_supports_sentence_pairs():
    dataset = Dataset.from_dict(
        {
            "question1": [
                "How do I learn Python?",
                "How can I study Python?",
                "What is the capital of France?",
                "Where is Paris located?",
            ],
            "question2": [
                "What is a good Python tutorial?",
                "Best way to learn programming?",
                "Which city is France capital?",
                "How to bake bread?",
            ],
            "label": [1, 1, 1, 0],
        }
    )

    subset = select_kcenter_tfidf(
        dataset,
        text_columns=("question1", "question2"),
        k_per_class=1,
        seed=7,
    )

    assert len(subset) == 2
    assert sorted(subset["label"]) == [0, 1]
