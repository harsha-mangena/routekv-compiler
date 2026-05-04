"""
TieringPlanner — assigns KV pages to memory tiers.

Tree-of-Thoughts branch: placement-first
----------------------------------------
The planner ranks pages by composite_score (route_score × 0.6 + hotness × 0.4),
then places the top fraction in HBM, the middle band in DRAM, and the tail
in storage-backed transform coding.

The tier budgets are configurable so experiments can test different split ratios.
"""
from .ir import KVPage, PlacementDecision, Tier, Encoding


TIER_ENCODING_MAP: dict[str, Encoding] = {
    "hbm": "fp16",
    "dram": "int8",
    "storage": "transform",
}


class TierBudgetRatio:
    """Fractional split of pages across tiers."""
    def __init__(self, hbm: float = 0.25, dram: float = 0.45, storage: float = 0.30):
        assert abs(hbm + dram + storage - 1.0) < 1e-6, "Ratios must sum to 1.0"
        self.hbm = hbm
        self.dram = dram
        self.storage = storage


class TieringPlanner:
    """
    Assigns each KV page to a tier based on its composite_score.

    Parameters
    ----------
    budget : TierBudgetRatio
        Fractional split of how many pages go to each tier.
    prefetch_fraction : float
        Fraction of hbm+dram pages that get proactive prefetch.
    """

    def __init__(
        self,
        budget: TierBudgetRatio | None = None,
        prefetch_fraction: float = 0.5,
    ):
        self.budget = budget or TierBudgetRatio()
        self.prefetch_fraction = prefetch_fraction

    def plan(self, pages: list[KVPage]) -> list[PlacementDecision]:
        if not pages:
            return []

        ranked = sorted(pages, key=lambda p: p.composite_score, reverse=True)
        n = len(ranked)
        hbm_cut = max(1, int(n * self.budget.hbm))
        dram_cut = max(hbm_cut + 1, int(n * (self.budget.hbm + self.budget.dram)))
        prefetch_cut = max(1, int(n * (self.budget.hbm + self.budget.dram) * self.prefetch_fraction))

        decisions: list[PlacementDecision] = []
        for idx, page in enumerate(ranked):
            if idx < hbm_cut:
                tier: Tier = "hbm"
            elif idx < dram_cut:
                tier = "dram"
            else:
                tier = "storage"

            decisions.append(
                PlacementDecision(
                    page_id=page.page_id,
                    tier=tier,
                    encoding=TIER_ENCODING_MAP[tier],
                    prefetch=idx < prefetch_cut,
                )
            )
        return decisions

    def summarise(self, decisions: list[PlacementDecision]) -> dict:
        from collections import Counter
        tiers = Counter(d.tier for d in decisions)
        prefetch_count = sum(1 for d in decisions if d.prefetch)
        return {"tier_counts": dict(tiers), "prefetch_count": prefetch_count, "total": len(decisions)}
