"""Collect per-run JSON artifacts into a single DataFrame.

`collect_runs` walks ``artifacts/runs/*`` and joins each run's ``config.json``
and ``metrics.json`` into one row. Nested ``timings`` are flattened into
``timings_<name>`` columns so analysis notebooks can plot them directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from text_distillation.utils import load_json


def collect_runs(runs_dir: str | Path = "artifacts/runs"):
    """Read all `runs_dir/*/{config.json,metrics.json}` into a `pandas.DataFrame`.

    Runs missing either file are skipped silently — partial runs (e.g. only
    config persisted, training crashed) do not produce a row.
    """
    import pandas as pd

    rows = list(iter_run_rows(runs_dir))
    return pd.DataFrame(rows)


def iter_run_rows(runs_dir: str | Path):
    """Yield one flat dict per completed run."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        if not config_path.exists() or not metrics_path.exists():
            continue
        config = load_json(config_path)
        metrics = load_json(metrics_path)
        yield _flatten_row(run_dir.name, config, metrics)


def _flatten_row(run_name: str, config: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"run_name": run_name}
    row.update({k: v for k, v in config.items() if not _is_complex(v)})
    method_kwargs = config.get("method_kwargs")
    if isinstance(method_kwargs, dict):
        for key, value in method_kwargs.items():
            if not _is_complex(value):
                row[f"method_kwargs_{key}"] = value

    for key, value in metrics.items():
        if key == "timings" and isinstance(value, dict):
            for tname, tval in value.items():
                row[f"timings_{tname}"] = tval
            continue
        if not _is_complex(value):
            row[key] = value

    if "full_train_size" in row and "k_total" in row and row.get("k_total"):
        row.setdefault("compression_ratio", row["full_train_size"] / row["k_total"])
    return row


def _is_complex(value: Any) -> bool:
    return isinstance(value, (dict, list, tuple, set))
