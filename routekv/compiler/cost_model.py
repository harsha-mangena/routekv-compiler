"""
CostModel — predicts the optimal tier+encoding for a KV page given hardware stats.

Atom: this is the learning core of the compiler.
In Phase 1 it ships as a rule-based heuristic.
In Phase 2 it will be replaced with a trained gradient-boosted model
or a small MLP that consumes trace features.
"""
from __future__ import annotations
from .ir import KVPage, Tier, Encoding


class CostModel:
    """
    Rule-based cost model (Phase 1 placeholder).

    Scoring logic:
      - composite_score >= hbm_threshold  -> hbm / fp16
      - composite_score >= dram_threshold -> dram / int8
      - else                              -> storage / transform

    The thresholds are tunable hyperparameters for sweep experiments.
    """

    def __init__(self, hbm_threshold: float = 0.65, dram_threshold: float = 0.30):
        self.hbm_threshold = hbm_threshold
        self.dram_threshold = dram_threshold

    def predict_tier(self, page: KVPage) -> tuple[Tier, Encoding]:
        score = page.composite_score
        if score >= self.hbm_threshold:
            return "hbm", "fp16"
        elif score >= self.dram_threshold:
            return "dram", "int8"
        else:
            return "storage", "transform"

    def batch_predict(self, pages: list[KVPage]) -> list[tuple[Tier, Encoding]]:
        return [self.predict_tier(p) for p in pages]
