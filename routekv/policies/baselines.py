"""
Baseline KV retention policies.

These are the baselines RouteKV aims to beat:
  lru_score       — rank by most-recent token position (pure recency)
  recency_policy  — apply lru_score to produce a ranked page list

All policies take a list[KVPage] and return a sorted list[KVPage]
with the most important pages first.
"""
from routekv.compiler.ir import KVPage


def lru_score(page: KVPage) -> float:
    """Score by recency only — higher end_token is more recent."""
    return float(page.end_token)


def recency_policy(pages: list[KVPage]) -> list[KVPage]:
    """Rank pages by LRU score descending (most recent first)."""
    return sorted(pages, key=lru_score, reverse=True)
