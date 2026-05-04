# RouteKV — Atom of Thoughts × Tree of Thoughts Analysis

## Atom Map

The KV cache problem decomposes into five orthogonal atoms.
Current systems typically optimise one or two; RouteKV targets all five jointly.

| Atom | What it decides | Current state of the art |
|------|----------------|-------------------------|
| Placement | Where KV lives (HBM/DRAM/storage) | ScoutAttention (2026), CPU offload (NVIDIA) |
| Representation | How each tier stores it (fp16/int8/int4/transform/landmark) | KVTC (2026), KVQuant, KIVI |
| Selection | Which pages stay hot, which get recalled lazily | ShadowKV, H2O, PyramidKV |
| Scheduling | When to prefetch, evict, compress, restore | ScoutAttention layer-ahead heuristic |
| Sharing | Cross-request/task/replica KV reuse | LMCache, llm-d KV routing |

## Tree of Thoughts — Three Main Branches

### Branch 1: Compression-first
- Compress aggressively at every tier
- Best paper: KVTC — 20x compression vs token eviction, quant, SVD
- Weakness: breaks retrieval-heavy tasks with aggressive low-rank keys

### Branch 2: Offload-first
- Keep recent/hot pages in HBM; move cold ones to DRAM/SSD
- Best paper: ScoutAttention (2026) — layer-ahead CPU pre-computation
- Weakness: fixed heuristics do not adapt to workload type

### Branch 3: Routing-first (RouteKV's branch)
- Model attention as a routing substrate
- Protect pages that lie on high-probability answer routes
- Basis: 'Physics of KV Compression' (2026) — route reachability is the real constraint
- Key insight: route_score should gate ALL other decisions

## Interrelated Dot: Why All Atoms Must Be Co-Optimised

Speculative decoding changes kernel shapes
→ which changes CUDA graph reuse
→ which changes KV page access patterns
→ which changes what should be HBM-resident vs DRAM
→ which changes whether transform coding or landmark summaries are a better encoding
→ which changes scheduling bandwidth requirements
→ which changes whether inter-request reuse is feasible

RouteKV closes this loop by feeding route_score and workload features
into a unified cost model that decides across all five atoms simultaneously.

## Key Papers

- KV Cache Optimization Strategies for Scalable LLM Inference (arXiv 2603.20397, 2026)
- KV Cache Offloading for Context-Intensive Tasks (arXiv 2604.08426, 2026)
- ScoutAttention: Layer-Ahead CPU Pre-computation (arXiv 2603.27138, 2026)
- KV Cache Transform Coding for Compact Storage (arXiv 2511.01815, 2025)
- Understanding the Physics of KV Cache Compression (arXiv 2603.01426, 2026)
- Sequential KV Cache Compression via Probabilistic Language Tries (arXiv 2604.15356, 2026)
- End-to-End Transformer Acceleration via Processing-in-Memory (arXiv 2601.14260, 2025)
