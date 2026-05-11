from text_distillation.evaluation import (
    compute_accuracy,
    compute_accuracy_f1_average,
    compute_classification_metrics,
    compute_f1_macro,
)


def test_compute_accuracy():
    assert compute_accuracy([0, 1, 1, 0], [0, 1, 0, 0]) == 0.75


def test_compute_f1_macro():
    score = compute_f1_macro([0, 1, 1, 0], [0, 1, 0, 0])

    assert round(score, 4) == 0.7333


def test_compute_classification_metrics():
    metrics = compute_classification_metrics([0, 1, 1, 0], [0, 1, 0, 0])

    assert metrics == {"accuracy": 0.75, "f1_macro": compute_f1_macro([0, 1, 1, 0], [0, 1, 0, 0])}


def test_compute_qqp_style_metric():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    metrics = compute_classification_metrics(y_true, y_pred, metric_name="accuracy_f1_average")

    assert metrics["accuracy"] == 0.75
    assert round(metrics["f1"], 4) == 0.6667
    assert metrics["score"] == compute_accuracy_f1_average(y_true, y_pred)
