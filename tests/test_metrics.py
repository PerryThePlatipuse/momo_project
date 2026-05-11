from text_distillation.evaluation import compute_accuracy, compute_classification_metrics, compute_f1_macro


def test_compute_accuracy():
    assert compute_accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_compute_f1_macro():
    score = compute_f1_macro([0, 1, 1, 0], [0, 1, 0, 0])

    assert round(score, 4) == 0.7333


def test_compute_classification_metrics():
    metrics = compute_classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert metrics == {"accuracy": 0.75, "f1_macro": compute_f1_macro([0, 1, 1, 0], [0, 1, 0, 0])}

