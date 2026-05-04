"""
TraceCollector — accumulates DecodeStepTrace records and computes workload features.

This is the entry point for Phase 1 benchmarking:
  1. Attach a collector to a decode loop.
  2. Call record() at each step.
  3. Call feature_summary() to get the feature vector that feeds the CostModel.
"""
from __future__ import annotations
import statistics
from .schema import DecodeStepTrace


class TraceCollector:
    def __init__(self):
        self._traces: list[DecodeStepTrace] = []

    def record(self, trace: DecodeStepTrace) -> None:
        self._traces.append(trace)

    @property
    def traces(self) -> list[DecodeStepTrace]:
        return list(self._traces)

    def feature_summary(self) -> dict:
        """Aggregate workload features for cost-model input."""
        if not self._traces:
            return {}
        densities = [t.attention_density for t in self._traces]
        gpu_stalls = [t.gpu_stall_us for t in self._traces]
        cpu_stalls = [t.cpu_stall_us for t in self._traces]
        return {
            "n_steps": len(self._traces),
            "mean_attention_density": round(statistics.mean(densities), 4),
            "mean_gpu_stall_us": round(statistics.mean(gpu_stalls), 3),
            "mean_cpu_stall_us": round(statistics.mean(cpu_stalls), 3),
            "max_seq_len": max(t.sequence_length for t in self._traces),
            "dominant_context_type": statistics.mode(t.context_intensity for t in self._traces),
        }

    def export_jsonl(self, path: str) -> None:
        import json
        with open(path, "w") as f:
            for t in self._traces:
                f.write(t.model_dump_json() + "\n")
