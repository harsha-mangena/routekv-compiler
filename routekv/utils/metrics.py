"""Lightweight metrics tracker for benchmarking."""
import time
from collections import defaultdict
from typing import Dict, List
import numpy as np


class MetricsTracker:
    """Accumulate timing + memory metrics across decode steps."""

    def __init__(self):
        self._data: Dict[str, List[float]] = defaultdict(list)
        self._timers: Dict[str, float] = {}

    def start(self, key: str) -> None:
        self._timers[key] = time.perf_counter()

    def stop(self, key: str) -> float:
        elapsed = time.perf_counter() - self._timers.pop(key, time.perf_counter())
        self._data[key].append(elapsed * 1000)  # ms
        return elapsed

    def record(self, key: str, value: float) -> None:
        self._data[key].append(value)

    def summary(self) -> Dict[str, Dict[str, float]]:
        return {
            k: {
                "mean": float(np.mean(v)),
                "p50": float(np.percentile(v, 50)),
                "p95": float(np.percentile(v, 95)),
                "p99": float(np.percentile(v, 99)),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for k, v in self._data.items()
        }

    def reset(self) -> None:
        self._data.clear()
        self._timers.clear()
