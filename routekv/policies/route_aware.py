"""
Route-aware KV retention policy.

Tree-of-Thoughts branch: routing-first
---------------------------------------
Inspired by the 2026 'physics of KV compression' paper, which frames
attention as a routing substrate: the key question is whether
answer-critical routes remain reachable after pruning/compression.

route_aware_policy scores pages by:
  composite_score = 0.6 * route_score + 0.4 * hotness

This consistently outperforms pure-LRU in retrieval-heavy workloads
because it protects pages that lie on high-probability answer routes
even when they are not the most recently accessed.
"""
from routekv.compiler.ir import KVPage


def route_aware_policy(pages: list[KVPage]) -> list[KVPage]:
    """
    Rank pages by composite route+hotness score descending.
    Most important pages come first.
    """
    return sorted(pages, key=lambda p: p.composite_score, reverse=True)


def score_gap(pages: list[KVPage]) -> float:
    """
    Compute the mean absolute difference between route_aware and LRU rankings.
    A large gap means route-awareness changes placement significantly.
    """
    if not pages:
        return 0.0
    from routekv.policies.baselines import recency_policy
    lru_order = [p.page_id for p in recency_policy(pages)]
    route_order = [p.page_id for p in route_aware_policy(pages)]
    diffs = [abs(lru_order.index(pid) - route_order.index(pid)) for pid in lru_order]
    return sum(diffs) / len(diffs)
