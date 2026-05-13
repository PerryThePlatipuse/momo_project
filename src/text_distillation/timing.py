"""Small wall-clock timing helper used across experiments.

`TimingTracker` accumulates named durations via a context manager and produces
a plain `dict[str, float]` that gets nested under ``"timings"`` in
``metrics.json``. Times are wall-clock seconds from ``time.perf_counter``.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class TimingTracker:
    def __init__(self) -> None:
        self._timings: dict[str, float] = {}

    @contextmanager
    def measure(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings[name] = self._timings.get(name, 0.0) + (
                time.perf_counter() - start
            )

    def add(self, name: str, seconds: float) -> None:
        self._timings[name] = self._timings.get(name, 0.0) + float(seconds)

    def merge(self, other: "TimingTracker") -> None:
        for name, seconds in other.as_dict().items():
            self.add(name, seconds)

    def as_dict(self) -> dict[str, float]:
        return dict(self._timings)

    def __contains__(self, name: str) -> bool:
        return name in self._timings

    def __getitem__(self, name: str) -> float:
        return self._timings[name]
