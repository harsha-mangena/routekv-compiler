# RouteKV Compiler — Roadmap

## Phase 1 — Offline Baseline (Colab-compatible)

| Task | Status |
|------|--------|
| KVPage / PlacementDecision IR | ✅ done |
| TieringPlanner (ratio-based) | ✅ done |
| CostModel stub (rule-based) | ✅ done |
| MemoryTierRuntime | ✅ done |
| KVPageSimulator (bandwidth model) | ✅ done |
| Baseline policies (LRU) | ✅ done |
| Route-aware policy | ✅ done |
| TraceCollector + schema | ✅ done |
| Unit tests | ✅ done |
| Baseline profiling notebook | 🔄 in progress |

## Phase 2 — Cost Model Training

- Collect traces from vLLM decode loops on Llama-3-8B / Mistral-7B
- Build feature matrix: attention_density, gpu_stall_us, cpu_stall_us, context_intensity, seq_len
- Train gradient-boosted or MLP cost model on tier placement labels
- Replace rule-based CostModel with trained predictor
- A/B compare: LRU vs route-aware vs learned policy

## Phase 3 — Adaptive Runtime

- Plug into vLLM PagedAttention block manager
- Add prefetch controller (layer-ahead, bandwidth-aware)
- Add inter-request KV reuse for semantically similar prompts
- Storage-backed cache federation across replicas
- Benchmark: TTFT, ITL, peak memory, quality (RULER, LongBench)

## Phase 4 — Futuristic Extensions

- Transform-coded KV (KVTC-style) integration
- Route-criticality scoring via attention rollout
- Neuromorphic context triage hook (long-context pre-filter)
- Photonic coprocessor abstraction stub
