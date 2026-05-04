"""
KVPageSimulator — offline discrete-step simulator for KV tiering experiments.

Purpose
-------
Before touching a real inference engine, we simulate:
  - decode steps advancing token-by-token
  - per-step tier hit/miss/eviction events
  - stall time estimation (bandwidth-based)

This feeds the cost model training loop in Phase 2.

Bandwidth assumptions (configurable in hardware.yaml):
  HBM   ~2.0  TB/s  (A100/H100 class)
  DRAM  ~0.05 TB/s  (CPU DDR5, PCIe-limited)
  SSD   ~0.007 TB/s (NVMe SSD, rough estimate)
"""
from __future__ import annotations
from dataclasses import dataclass
from routekv.compiler.ir import KVPage, PlacementDecision


BW_BYTES_PER_US = {
    "hbm": 2_000_000,     # 2 TB/s  -> 2_000_000 B/µs
    "dram": 50_000,       # 50 GB/s -> 50_000 B/µs
    "storage": 7_000,     # 7 GB/s  -> 7_000 B/µs
}


@dataclass
class SimStepResult:
    step: int
    hbm_hits: int
    dram_hits: int
    storage_hits: int
    estimated_stall_us: float
    total_bytes_moved: int


class KVPageSimulator:
    """
    Simulates one decode pass given a fixed set of pages and placement decisions.

    Parameters
    ----------
    pages       : list of KVPage
    decisions   : list of PlacementDecision (one per page, matched by page_id)
    n_steps     : number of decode steps to simulate
    """

    def __init__(
        self,
        pages: list[KVPage],
        decisions: list[PlacementDecision],
        n_steps: int = 128,
    ):
        self.pages = {p.page_id: p for p in pages}
        self.placement: dict[str, str] = {d.page_id: d.tier for d in decisions}
        self.n_steps = n_steps

    def run(self) -> list[SimStepResult]:
        results: list[SimStepResult] = []
        for step in range(self.n_steps):
            hbm = dram = storage = 0
            total_bytes = 0
            total_stall = 0.0
            for pid, tier in self.placement.items():
                page = self.pages.get(pid)
                b = page.bytes_estimate if page else 0
                if tier == "hbm":
                    hbm += 1
                elif tier == "dram":
                    dram += 1
                else:
                    storage += 1
                total_bytes += b
                total_stall += b / BW_BYTES_PER_US.get(tier, 1)
            results.append(SimStepResult(
                step=step,
                hbm_hits=hbm,
                dram_hits=dram,
                storage_hits=storage,
                estimated_stall_us=round(total_stall, 3),
                total_bytes_moved=total_bytes,
            ))
        return results

    def summary(self, results: list[SimStepResult]) -> dict:
        avg_stall = sum(r.estimated_stall_us for r in results) / len(results)
        return {
            "n_steps": self.n_steps,
            "avg_stall_us": round(avg_stall, 3),
            "total_bytes_moved": sum(r.total_bytes_moved for r in results),
        }
