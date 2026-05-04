from routekv.profiler.schema import DecodeStepTrace
from routekv.profiler.collector import TraceCollector


def make_trace(step: int) -> DecodeStepTrace:
    return DecodeStepTrace(
        step=step,
        sequence_length=1024 + step * 4,
        layer=step % 32,
        attention_density=0.6,
        kv_bytes_touched=1024 * 64,
        gpu_stall_us=12.5,
        cpu_stall_us=3.2,
        context_intensity="retrieval",
    )


def test_collector_records_traces():
    c = TraceCollector()
    for i in range(10):
        c.record(make_trace(i))
    assert len(c.traces) == 10


def test_feature_summary_keys():
    c = TraceCollector()
    for i in range(20):
        c.record(make_trace(i))
    summary = c.feature_summary()
    assert "mean_attention_density" in summary
    assert "dominant_context_type" in summary
    assert summary["dominant_context_type"] == "retrieval"
