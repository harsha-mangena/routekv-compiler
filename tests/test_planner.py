import pytest
from routekv.compiler.ir import KVPage
from routekv.compiler.planner import TieringPlanner, TierBudgetRatio


def make_pages(n: int, base_score: float = 0.5) -> list[KVPage]:
    return [
        KVPage(
            layer=i % 32,
            head_group=0,
            start_token=i * 128,
            end_token=(i + 1) * 128,
            bytes_estimate=1024 * 64,
            hotness=round((n - i) / n, 3),
            route_score=round((n - i) / n * base_score, 3),
        )
        for i in range(n)
    ]


def test_planner_places_hottest_pages_in_hbm():
    planner = TieringPlanner()
    pages = make_pages(32)
    decisions = planner.plan(pages)
    assert decisions[0].tier == "hbm"


def test_planner_places_coldest_pages_in_storage():
    planner = TieringPlanner()
    pages = make_pages(40)
    decisions = planner.plan(pages)
    assert any(d.tier == "storage" for d in decisions)


def test_planner_encoding_matches_tier():
    planner = TieringPlanner()
    pages = make_pages(40)
    decisions = planner.plan(pages)
    enc_map = {"hbm": "fp16", "dram": "int8", "storage": "transform"}
    for d in decisions:
        assert d.encoding == enc_map[d.tier]


def test_planner_summary():
    planner = TieringPlanner(budget=TierBudgetRatio(hbm=0.25, dram=0.45, storage=0.30))
    pages = make_pages(100)
    decisions = planner.plan(pages)
    summary = planner.summarise(decisions)
    assert summary["total"] == 100
    assert "tier_counts" in summary


def test_empty_pages_returns_empty_decisions():
    planner = TieringPlanner()
    assert planner.plan([]) == []
