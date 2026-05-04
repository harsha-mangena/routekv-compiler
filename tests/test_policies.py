from routekv.compiler.ir import KVPage
from routekv.policies.baselines import recency_policy
from routekv.policies.route_aware import route_aware_policy, score_gap


def make_pages() -> list[KVPage]:
    return [
        KVPage(layer=0, head_group=0, start_token=0, end_token=64, bytes_estimate=1024, hotness=0.9, route_score=0.1),
        KVPage(layer=1, head_group=0, start_token=64, end_token=128, bytes_estimate=1024, hotness=0.3, route_score=0.95),
        KVPage(layer=2, head_group=0, start_token=128, end_token=192, bytes_estimate=1024, hotness=0.5, route_score=0.5),
    ]


def test_lru_prefers_latest_token():
    pages = make_pages()
    ranked = recency_policy(pages)
    assert ranked[0].end_token == max(p.end_token for p in pages)


def test_route_aware_prefers_high_route_score():
    pages = make_pages()
    ranked = route_aware_policy(pages)
    assert ranked[0].route_score == max(p.route_score for p in pages)


def test_score_gap_is_non_negative():
    pages = make_pages()
    assert score_gap(pages) >= 0.0
