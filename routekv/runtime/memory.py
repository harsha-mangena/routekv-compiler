"""
MemoryTierRuntime — abstract memory hierarchy for RouteKV.

Models a three-tier memory system:
  HBM      — on-GPU high-bandwidth memory (fast, limited)
  DRAM     — CPU system memory (medium speed, large)
  Storage  — SSD-backed or network-attached (slow, vast)

In real integration this wraps torch CUDA allocators and
cpuoffload tensors. In Phase 1 it tracks byte budgets.
"""
from dataclasses import dataclass, field
from routekv.compiler.ir import PlacementDecision


@dataclass
class TierBudget:
    hbm_bytes: int
    dram_bytes: int
    storage_bytes: int


@dataclass
class TierState:
    used_bytes: int = 0
    page_ids: list[str] = field(default_factory=list)


class MemoryTierRuntime:
    """Tracks live byte usage across tiers given a PlacementDecision list."""

    def __init__(self, budget: TierBudget):
        self.budget = budget
        self._state: dict[str, TierState] = {
            "hbm": TierState(),
            "dram": TierState(),
            "storage": TierState(),
        }

    def apply(self, decisions: list[PlacementDecision], page_bytes: dict[str, int]) -> None:
        """Apply a set of placement decisions, updating tier usage."""
        for d in decisions:
            state = self._state[d.tier]
            state.page_ids.append(d.page_id)
            state.used_bytes += page_bytes.get(d.page_id, 0)

    def utilisation(self) -> dict[str, float]:
        """Return fraction of budget used per tier."""
        caps = {
            "hbm": self.budget.hbm_bytes,
            "dram": self.budget.dram_bytes,
            "storage": self.budget.storage_bytes,
        }
        return {
            tier: (self._state[tier].used_bytes / caps[tier]) if caps[tier] > 0 else 0.0
            for tier in caps
        }

    def describe(self) -> dict:
        return {
            "budget": {
                "hbm_bytes": self.budget.hbm_bytes,
                "dram_bytes": self.budget.dram_bytes,
                "storage_bytes": self.budget.storage_bytes,
            },
            "utilisation": self.utilisation(),
        }
