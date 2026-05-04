from routekv.compiler.ir import KVPage
from routekv.compiler.planner import TieringPlanner
from routekv.runtime.simulator import KVPageSimulator


def make_pages(n: int) -> list[KVPage]:
    return [
        KVPage(
            layer=i % 32, head_group=0,
            start_token=i * 64, end_token=(i + 1) * 64,
            bytes_estimate=1024 * 32,
            hotness=round((n - i) / n, 3),
            route_score=round((n - i) / n * 0.8, 3),
        )
        for i in range(n)
    ]


def test_simulator_runs_all_steps():
    pages = make_pages(20)
    planner = TieringPlanner()
    decisions = planner.plan(pages)
    sim = KVPageSimulator(pages=pages, decisions=decisions, n_steps=16)
    results = sim.run()
    assert len(results) == 16


def test_simulator_stall_positive():
    pages = make_pages(20)
    decisions = TieringPlanner().plan(pages)
    sim = KVPageSimulator(pages=pages, decisions=decisions, n_steps=8)
    results = sim.run()
    assert all(r.estimated_stall_us > 0 for r in results)


def test_simulator_summary_keys():
    pages = make_pages(20)
    decisions = TieringPlanner().plan(pages)
    sim = KVPageSimulator(pages=pages, decisions=decisions, n_steps=8)
    summary = sim.summary(sim.run())
    assert "avg_stall_us" in summary
    assert "total_bytes_moved" in summary
