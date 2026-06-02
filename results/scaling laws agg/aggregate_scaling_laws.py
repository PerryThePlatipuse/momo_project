#!/usr/bin/env python3
"""Aggregate available scaling-law results and draw one SVG subplot grid.

The script intentionally uses only the Python standard library. The current
repo environment may not have pandas/matplotlib installed, while the result
tables are simple CSV files.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

OUT_AGG = SCRIPT_DIR / "scaling_laws_aggregated.csv"
OUT_COVERAGE = SCRIPT_DIR / "scaling_laws_coverage.csv"
OUT_MISSING = SCRIPT_DIR / "scaling_laws_missing_grid.csv"
OUT_SVG = SCRIPT_DIR / "scaling_laws_all_subplots.svg"

CORESET_SOURCES = [
    REPO_ROOT / "results" / "scaling_laws" / "scaling_laws_all_methods.csv",
    REPO_ROOT / "scaling_laws_coreset_report" / "scaling_laws_all_methods.csv",
]
HERDING_SOURCE = REPO_ROOT / "scaling_laws_herding_report" / "raw_results.csv"
EMBEDDING_SOURCE = REPO_ROOT / "results" / "emdebding_distilation" / "metrics.csv"

TARGET_METHODS = ["herding", "random", "k_centers", "dilm"]
METHOD_ORDER = ["random", "k_centers", "herding", "embedding_distillation", "dilm"]
METHOD_COLORS = {
    "random": "#4e79a7",
    "k_centers": "#f28e2b",
    "herding": "#59a14f",
    "embedding_distillation": "#b07aa1",
    "dilm": "#e15759",
}

DPC_ORDER = [1, 5, 10, 20, 50, 100, 500, 1000]
AGG_COLUMNS = [
    "method",
    "task",
    "learner",
    "dpc",
    "k",
    "num_synthetic_examples",
    "metric",
    "score",
    "score_std",
    "accuracy_mean",
    "accuracy_std",
    "f1_mean",
    "f1_std",
    "macro_f1_mean",
    "macro_f1_std",
    "n_dataset",
    "seed",
    "selection_time_sec",
    "eval_time_sec",
    "elapsed_sec_mean",
    "runs",
    "source_path",
]


def read_csv(path: Path, **kwargs) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding=kwargs.pop("encoding", "utf-8-sig")) as f:
        return list(csv.DictReader(f, **kwargs))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def blank_row(source_path: Path) -> dict[str, str]:
    row = {key: "" for key in AGG_COLUMNS}
    row["source_path"] = str(source_path.relative_to(REPO_ROOT))
    return row


def normalize_coreset(path: Path) -> list[dict[str, str]]:
    rows = []
    for raw in read_csv(path):
        row = blank_row(path)
        row.update(
            {
                "method": raw["method"],
                "task": raw["task"],
                "learner": raw["learner"],
                "dpc": raw["dpc"],
                "k": raw["k"],
                "num_synthetic_examples": raw["k"],
                "metric": "accuracy",
                "score": raw["score"],
                "accuracy_mean": raw["score"],
            }
        )
        rows.append(row)
    return rows


def normalize_herding(path: Path) -> list[dict[str, str]]:
    rows = []
    for raw in read_csv(path):
        row = blank_row(path)
        row.update(
            {
                "method": raw["method"],
                "task": raw["task"],
                "learner": raw["learner"],
                "dpc": raw["dpc"],
                "k": raw["k"],
                "num_synthetic_examples": raw["k"],
                "metric": raw["metric"],
                "score": raw["score"],
                "score_std": raw["score_std"],
                "accuracy_mean": raw.get("accuracy_mean", ""),
                "accuracy_std": raw.get("accuracy_std", ""),
                "f1_mean": raw.get("combined_score_mean", ""),
                "f1_std": raw.get("combined_score_std", ""),
                "n_dataset": raw.get("n_dataset", ""),
                "seed": raw.get("seed", ""),
                "selection_time_sec": raw.get("selection_time_sec", ""),
                "eval_time_sec": raw.get("eval_time_sec", ""),
            }
        )
        rows.append(row)
    return rows


def normalize_embedding(path: Path) -> list[dict[str, str]]:
    rows = []
    for raw in read_csv(path, delimiter=";"):
        base = blank_row(path)
        base.update(
            {
                "method": "embedding_distillation",
                "task": "ag_news",
                "learner": "bert-base-uncased",
                "dpc": raw["dpc"],
                "k": raw["num_synthetic_examples"],
                "num_synthetic_examples": raw["num_synthetic_examples"],
                "accuracy_mean": raw["accuracy_mean"],
                "accuracy_std": raw["accuracy_std"],
                "macro_f1_mean": raw["macro_f1_mean"],
                "macro_f1_std": raw["macro_f1_std"],
                "elapsed_sec_mean": raw["elapsed_sec_mean"],
                "runs": raw["runs"],
            }
        )
        accuracy = dict(base)
        accuracy.update(
            {
                "metric": "accuracy",
                "score": raw["accuracy_mean"],
                "score_std": raw["accuracy_std"],
            }
        )
        macro_f1 = dict(base)
        macro_f1.update(
            {
                "metric": "macro_f1",
                "score": raw["macro_f1_mean"],
                "score_std": raw["macro_f1_std"],
            }
        )
        rows.extend([accuracy, macro_f1])
    return rows


def as_int(value: str) -> int:
    return int(float(value))


def as_float(value: str) -> float | None:
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def sort_key(row: dict[str, str]) -> tuple:
    method_rank = METHOD_ORDER.index(row["method"]) if row["method"] in METHOD_ORDER else 99
    dpc = as_int(row["dpc"]) if row["dpc"] else 0
    return (method_rank, row["task"], row["learner"], dpc, row["metric"])


def collect_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in CORESET_SOURCES:
        rows.extend(normalize_coreset(source))
    rows.extend(normalize_herding(HERDING_SOURCE))
    rows.extend(normalize_embedding(EMBEDDING_SOURCE))

    # The coreset CSV exists both under results/ and as a report copy. Keep one
    # normalized point per method/task/learner/dpc/metric/score.
    seen: set[tuple[str, ...]] = set()
    deduped = []
    for row in rows:
        key = (
            row["method"],
            row["task"],
            row["learner"],
            row["dpc"],
            row["metric"],
            row["score"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return sorted(deduped, key=sort_key)


def primary_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [r for r in rows if r["metric"] in {"accuracy", "combined_score"}]


def build_coverage(rows: list[dict[str, str]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    primary = primary_rows(rows)
    herding = [r for r in primary if r["method"] == "herding"]
    tasks = sorted({r["task"] for r in herding})
    learners = sorted({r["learner"] for r in herding})
    dpcs = sorted({as_int(r["dpc"]) for r in herding})
    expected = {(task, learner, dpc) for task in tasks for learner in learners for dpc in dpcs}

    coverage = []
    missing = []
    for method in TARGET_METHODS:
        actual = {
            (r["task"], r["learner"], as_int(r["dpc"]))
            for r in primary
            if r["method"] == method
        }
        missed = sorted(expected - actual)
        for task, learner, dpc in missed:
            missing.append({"method": method, "task": task, "learner": learner, "dpc": dpc})
        present_tasks = sorted({x[0] for x in actual})
        present_learners = sorted({x[1] for x in actual})
        present_dpcs = sorted({x[2] for x in actual})
        ratio = len(actual) / len(expected) if expected else 0.0
        if len(actual) == 0:
            status = "missing"
        elif len(actual) == len(expected):
            status = "complete"
        else:
            status = "partial"
        coverage.append(
            {
                "method": method,
                "status": status,
                "actual_points": len(actual),
                "expected_points": len(expected),
                "coverage_ratio": f"{ratio:.3f}",
                "tasks_present": "|".join(present_tasks),
                "learners_present": "|".join(present_learners),
                "dpcs_present": "|".join(str(x) for x in present_dpcs),
                "missing_tasks": "|".join(sorted(set(tasks) - set(present_tasks))),
                "missing_learners": "|".join(sorted(set(learners) - set(present_learners))),
            }
        )

    embedding_points = [
        (r["task"], r["learner"], as_int(r["dpc"]), r["metric"])
        for r in rows
        if r["method"] == "embedding_distillation"
    ]
    coverage.append(
        {
            "method": "embedding_distillation",
            "status": "extra_method",
            "actual_points": len(set(embedding_points)),
            "expected_points": "",
            "coverage_ratio": "",
            "tasks_present": "ag_news",
            "learners_present": "bert-base-uncased",
            "dpcs_present": "1|2|3|5|10|20",
            "missing_tasks": "",
            "missing_learners": "",
        }
    )
    return coverage, missing


def nice_label(text: str) -> str:
    return text.replace("microsoft/", "").replace("_", " ")


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" '
        f'fill="#222">{escape(text)}</text>'
    )


def draw_svg(rows: list[dict[str, str]]) -> None:
    plot_rows = primary_rows(rows)
    tasks = sorted({r["task"] for r in plot_rows}, key=lambda x: ["ag_news", "sst2", "mnli", "qqp"].index(x) if x in ["ag_news", "sst2", "mnli", "qqp"] else 99)
    learners = sorted({r["learner"] for r in plot_rows})

    panel_w, panel_h = 310, 230
    left, top = 135, 115
    gap_x, gap_y = 42, 64
    right, bottom = 235, 80
    width = left + len(learners) * panel_w + (len(learners) - 1) * gap_x + right
    height = top + len(tasks) * panel_h + (len(tasks) - 1) * gap_y + bottom

    grouped: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in plot_rows:
        score = as_float(row["score"])
        if score is None:
            continue
        grouped[(row["task"], row["learner"], row["method"])].append((as_int(row["dpc"]), score))

    all_scores = [score for values in grouped.values() for _, score in values]
    y_min = max(0.0, math.floor((min(all_scores) - 0.03) * 20) / 20) if all_scores else 0.0
    y_max = min(1.0, math.ceil((max(all_scores) + 0.03) * 20) / 20) if all_scores else 1.0
    if y_max - y_min < 0.2:
        y_min, y_max = max(0.0, y_min - 0.1), min(1.0, y_max + 0.1)

    x_min, x_max = math.log10(1), math.log10(1000)
    pad_l, pad_r, pad_t, pad_b = 38, 12, 18, 34
    inner_w = panel_w - pad_l - pad_r
    inner_h = panel_h - pad_t - pad_b

    def sx(dpc: int, px: float) -> float:
        return px + pad_l + (math.log10(dpc) - x_min) / (x_max - x_min) * inner_w

    def sy(score: float, py: float) -> float:
        return py + pad_t + (y_max - score) / (y_max - y_min) * inner_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(width / 2, 32, "Scaling laws: all available results", 22, "middle", "700"),
        svg_text(width / 2, 56, "x = DPC (log scale), y = primary metric: accuracy except QQP combined_score", 13, "middle"),
    ]

    legend_x = width - right + 25
    legend_y = top
    parts.append(svg_text(legend_x, legend_y - 22, "Legend", 13, "start", "700"))
    for i, method in enumerate(METHOD_ORDER):
        if method == "dilm":
            label = "dilm (no metrics)"
        else:
            label = method
        color = METHOD_COLORS[method]
        y = legend_y + i * 24
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 24}" y2="{y}" stroke="{color}" stroke-width="3"/>')
        parts.append(svg_text(legend_x + 32, y + 4, label, 13))

    y_ticks = [round(y_min + i * (y_max - y_min) / 4, 2) for i in range(5)]
    x_ticks = [1, 10, 100, 1000]

    for col, learner in enumerate(learners):
        px = left + col * (panel_w + gap_x)
        parts.append(svg_text(px + panel_w / 2, top - 18, nice_label(learner), 12, "middle", "700"))

    for row_idx, task in enumerate(tasks):
        py = top + row_idx * (panel_h + gap_y)
        parts.append(svg_text(24, py + panel_h / 2, nice_label(task), 14, "start", "700"))
        for col, learner in enumerate(learners):
            px = left + col * (panel_w + gap_x)
            parts.append(f'<rect x="{px}" y="{py}" width="{panel_w}" height="{panel_h}" fill="#fbfbfb" stroke="#d8d8d8"/>')
            for tick in y_ticks:
                y = sy(tick, py)
                parts.append(f'<line x1="{px + pad_l}" y1="{y:.1f}" x2="{px + panel_w - pad_r}" y2="{y:.1f}" stroke="#eeeeee"/>')
                if col == 0:
                    parts.append(svg_text(px + pad_l - 7, y + 4, f"{tick:.2f}", 10, "end"))
            for tick in x_ticks:
                x = sx(tick, px)
                parts.append(f'<line x1="{x:.1f}" y1="{py + pad_t}" x2="{x:.1f}" y2="{py + panel_h - pad_b}" stroke="#f1f1f1"/>')
                parts.append(svg_text(x, py + panel_h - 11, str(tick), 10, "middle"))
            parts.append(f'<line x1="{px + pad_l}" y1="{py + panel_h - pad_b}" x2="{px + panel_w - pad_r}" y2="{py + panel_h - pad_b}" stroke="#333"/>')
            parts.append(f'<line x1="{px + pad_l}" y1="{py + pad_t}" x2="{px + pad_l}" y2="{py + panel_h - pad_b}" stroke="#333"/>')

            for method in METHOD_ORDER:
                points = sorted(grouped.get((task, learner, method), []))
                if not points:
                    continue
                color = METHOD_COLORS[method]
                coords = [(sx(dpc, px), sy(score, py)) for dpc, score in points]
                if len(coords) > 1:
                    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
                    parts.append(f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2"/>')
                for x, y in coords:
                    parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{color}" stroke="white" stroke-width="1"/>')

    parts.append("</svg>")
    OUT_SVG.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    rows = collect_rows()
    coverage, missing = build_coverage(rows)
    write_csv(OUT_AGG, rows, AGG_COLUMNS)
    write_csv(
        OUT_COVERAGE,
        coverage,
        [
            "method",
            "status",
            "actual_points",
            "expected_points",
            "coverage_ratio",
            "tasks_present",
            "learners_present",
            "dpcs_present",
            "missing_tasks",
            "missing_learners",
        ],
    )
    write_csv(OUT_MISSING, missing, ["method", "task", "learner", "dpc"])
    draw_svg(rows)
    print(f"wrote {OUT_AGG.relative_to(REPO_ROOT)} ({len(rows)} rows)")
    print(f"wrote {OUT_COVERAGE.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_MISSING.relative_to(REPO_ROOT)} ({len(missing)} missing target points)")
    print(f"wrote {OUT_SVG.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
