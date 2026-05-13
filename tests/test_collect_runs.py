from text_distillation.analysis import collect_runs, iter_run_rows
from text_distillation.utils import save_json


def _write_run(runs_dir, name, config, metrics):
    run_dir = runs_dir / name
    run_dir.mkdir(parents=True)
    save_json(config, run_dir / "config.json")
    save_json(metrics, run_dir / "metrics.json")


def test_collect_runs_flattens_timings_and_method_kwargs(tmp_path):
    _write_run(
        tmp_path,
        "run_a",
        config={
            "experiment_name": "run_a",
            "method_name": "random",
            "model_name": "bert-base-uncased",
            "dataset_name": "ag_news",
            "k_total": 100,
            "full_train_size": 1000,
            "method_kwargs": {"max_features": 50000},
        },
        metrics={
            "train_loss": 0.5,
            "accuracy": 0.8,
            "f1_macro": 0.78,
            "timings": {"selection_sec": 1.2, "training_sec": 3.4},
        },
    )
    _write_run(
        tmp_path,
        "run_b",
        config={
            "experiment_name": "run_b",
            "method_name": "kcenter_cls",
            "model_name": "roberta-base",
            "dataset_name": "sst2",
            "k_total": 50,
            "full_train_size": 500,
        },
        metrics={
            "train_loss": 0.4,
            "accuracy": 0.85,
            "f1_macro": 0.84,
            "timings": {"selection_sec": 2.1, "training_sec": 5.0, "embedding_forward_sec": 0.7},
        },
    )

    df = collect_runs(tmp_path)

    assert len(df) == 2
    expected_cols = {
        "experiment_name",
        "method_name",
        "model_name",
        "dataset_name",
        "k_total",
        "accuracy",
        "f1_macro",
        "timings_selection_sec",
        "timings_training_sec",
        "compression_ratio",
        "method_kwargs_max_features",
    }
    assert expected_cols.issubset(set(df.columns))

    row_a = df[df["experiment_name"] == "run_a"].iloc[0]
    assert row_a["compression_ratio"] == 10.0
    assert row_a["method_kwargs_max_features"] == 50000


def test_collect_runs_skips_incomplete_runs(tmp_path):
    # Only config exists — must be skipped.
    incomplete = tmp_path / "broken"
    incomplete.mkdir()
    save_json({"experiment_name": "broken"}, incomplete / "config.json")

    rows = list(iter_run_rows(tmp_path))
    assert rows == []
