"""DiLM distilled-dataset loader.

Maekawa et al. «DiLM: Distilling Dataset into Language Model» (NAACL Findings 2024)
ship pre-generated synthetic datasets in `DiLM-main/DiLM-synthetic-data/`. The
generator was trained for 80k steps with the standard causal-LM loss, then
fine-tuned for 20k steps with their gradient-matching objective, then sampled
20 different `distilled` subsets per (task × DPC) combination.

This module loads those JSONL files and returns a regular `datasets.Dataset`
that fits straight into the project pipeline — no generator training required.

Supported tasks and sizes (from the shipped data, DiLM Table 1):
- `sst2`, `qqp`, `mnli` (alias: `mnli-m`)
- `k_per_class` (DPC) ∈ {5, 10, 20}
- `dataset_index` ∈ 0..19 — DiLM averages over 20 distilled datasets per row of
  their tables. Pick one per run, or sweep all 20 for the matching-paper-protocol
  experiment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from text_distillation.distillation import register_selection
from text_distillation.timing import TimingTracker


# Map our project's dataset names -> DiLM-main subdirectory names.
_TASK_DIR = {
    "sst2": "sst2",
    "qqp": "qqp",
    "mnli": "mnli",
    "mnli-m": "mnli",
}

# Mapping from DiLM field names -> project column names (per dataset).
# DiLM uses `labels`; the project uses `label`. Text columns match by construction.
_FIELD_RENAMES = {
    "sst2": {"labels": "label"},
    "qqp": {"labels": "label"},
    "mnli": {"labels": "label"},
    "mnli-m": {"labels": "label"},
}


@register_selection("dilm")
def distill_dilm(
    dataset: Any,
    dataset_name: str,
    k_per_class: int = 20,
    seed: int = 42,
    label_column: str = "label",
    dataset_index: int = 0,
    dilm_data_root: str | Path | None = None,
    tracker: TimingTracker | None = None,
    **_unused,
) -> Any:
    """Load one of the 20 pre-generated DiLM synthetic datasets.

    `dataset` is accepted for signature parity with other `select_*`
    functions and is otherwise unused: DiLM data is fully synthetic.
    """
    if dataset_name not in _TASK_DIR:
        raise ValueError(
            f"DiLM data is only shipped for {sorted(_TASK_DIR)}; got {dataset_name!r}"
        )
    if k_per_class not in {5, 10, 20}:
        raise ValueError(
            f"DiLM data is only shipped for DPC in {{5, 10, 20}}; got {k_per_class}"
        )
    if dataset_index < 0 or dataset_index > 19:
        raise ValueError(f"dataset_index must be in 0..19, got {dataset_index}")

    root = Path(dilm_data_root) if dilm_data_root is not None else _default_data_root()
    task_dir = root / _TASK_DIR[dataset_name] / "dilm.dc"
    if not task_dir.exists():
        raise FileNotFoundError(
            f"DiLM synthetic data not found at {task_dir}. "
            "Pass dilm_data_root=... or copy DiLM-main/DiLM-synthetic-data/ next to src/."
        )

    run_dir = _find_run_dir(task_dir, k_per_class)
    jsonl_path = run_dir / "dataset" / f"dataset_{dataset_index}.json"
    if not jsonl_path.exists():
        raise FileNotFoundError(f"DiLM dataset file not found: {jsonl_path}")

    _measure = tracker.measure if tracker is not None else _noop_context
    with _measure("dilm_load_sec"):
        records = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    renames = _FIELD_RENAMES[dataset_name]
    target_label = label_column
    normalized: dict[str, list] = {}
    for record in records:
        for k, v in record.items():
            out_key = renames.get(k, k)
            if out_key == "label":
                out_key = target_label
            normalized.setdefault(out_key, []).append(v)

    return Dataset.from_dict(normalized)


def _default_data_root() -> Path:
    """Locate DiLM-main/DiLM-synthetic-data relative to this file."""
    here = Path(__file__).resolve()
    # src/text_distillation/dilm.py -> parents[2] is the project root.
    candidate = here.parents[2] / "DiLM-main" / "DiLM-synthetic-data"
    return candidate


def _find_run_dir(task_dir: Path, k_per_class: int) -> Path:
    matches = sorted(task_dir.glob(f"dpc_{k_per_class}.*"))
    if not matches:
        raise FileNotFoundError(
            f"No DiLM run directory matching `dpc_{k_per_class}.*` under {task_dir}"
        )
    if len(matches) > 1:
        # Multiple runs for the same DPC is unexpected. Pick the first one
        # deterministically (sorted) and surface a warning via the path itself.
        pass
    return matches[0]


class _noop_context:
    def __init__(self, *_a, **_kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False
