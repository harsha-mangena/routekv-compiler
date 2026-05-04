# 🧠 RouteKV Compiler

> **A route-aware, context-sensitive KV cache memory tiering compiler for long-context LLM inference.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Research](https://img.shields.io/badge/Status-Research%20Prototype-orange)]()

---

## 🔬 Motivation

LLM inference is often memory‑bound at low concurrency — the dominant bottleneck is **moving KV tensors between HBM, CPU DRAM, and the attention cores**, not raw compute.

Most existing systems treat KV placement and movement as an afterthought:

- **vLLM** pages KV blocks with LRU eviction, largely workload‑agnostic.
- **SGLang RadixAttention** handles prefix sharing but not per‑layer tiering.
- **ScoutAttention** hides CPU latency with layer‑ahead prefetch but relies on fixed heuristics.
- **KVTC / LAVa / MixKV** compress or evict at a single stage, without coordinating with downstream routing decisions.

**RouteKV’s goal:** jointly decide *where* KV lives, *how* it is encoded, *when* to move it, and *which requests share it* — all together, at runtime, driven by workload context.

---

## 🏗️ Architecture Overview

High‑level view of the RouteKV Compiler stack:

![RouteKV architecture](./docs/routekv-architecture.png)

```text
┌─────────────────────────────────────────────────────────────────┐
│                      RouteKV Compiler Stack                     │
│                                                                 │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Profiler    │──▶│ Cost Model   │──▶│  Tier Plan Compiler  │ │
│  │  (offline)   │   │  (learned)   │   │  (online / JIT)      │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
│          │                 │                      │             │
│          ▼                 ▼                      ▼             │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                RouteKV Runtime Engine                     │ │
│  │                                                           │ │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────────┐ ┌──────────┐ │ │
│  │  │  HBM    │  │ CPU DRAM │  │ Transformed  │ │  Shared  │ │ │
│  │  │  Pool   │  │  Pool    │  │   Store      │ │ KV Store │ │ │
│  │  │ (hot)   │  │ (warm)   │  │  (cold)      │ │ (reuse)  │ │ │
│  │  └─────────┘  └──────────┘  └──────────────┘ └──────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
│          │                                                    │
│          ▼                                                    │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │        Attention Kernel Dispatcher (CUDA + Triton)        │ │
│  │  - Tier-aware attention (HBM-only / GPU-CPU co-attention) │ │
│  │  - Block-wise sparse scoring                              │ │
│  │  - Asynchronous PCIe transfer streams                     │ │
│  └───────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📦 Repository Structure

```text
routekv-compiler/
├── routekv/                    # Core library
│   ├── profiler/               # Offline trace collection
│   ├── cost_model/             # Learned tier assignment model
│   ├── compiler/               # Tier plan compiler (online/JIT)
│   ├── runtime/                # Memory tier pools + dispatcher
│   ├── kernels/                # CUDA/Triton attention kernels
│   └── scoring/                # KV importance + diversity scoring
├── configs/                    # YAML experiment configs
├── experiments/                # Benchmark scripts
├── notebooks/                  # Colab-ready research notebooks
├── tests/                      # Unit + integration tests
└── docs/                       # Architecture docs
```

---

## 🚀 Quick Start

> **Note:** This is a research prototype. APIs and configuration files are subject to change.

```bash
git clone https://github.com/harsha-mangena/routekv-compiler
cd routekv-compiler
pip install -e ".[dev]"
```

### 1. Collect profiling traces

```bash
python -m routekv.profiler.collect \
  --model meta-llama/Llama-3.2-1B \
  --dataset longbench \
  --output traces/llama_1b_traces.pkl
```

### 2. Train a cost model

```bash
python -m routekv.cost_model.train \
  --traces traces/llama_1b_traces.pkl \
  --output models/cost_model.pt
```

### 3. Run a tiering benchmark

```bash
python experiments/benchmark_tiering.py \
  --config configs/llama_1b_tier3.yaml
```

This pipeline:

1. Records KV traffic and attention statistics for a target model.
2. Trains a learned cost model that predicts optimal tier assignments.
3. Executes the model with RouteKV-controlled tiering and reports throughput / memory trade-offs.

---

## 💡 Usage Examples

These examples are **illustrative** and meant to show how RouteKV could be wired into an LLM stack; concrete APIs may evolve.

### Example 1 — Using RouteKV as a drop-in runtime wrapper

```python
# conceptual example API; subject to change
from routekv.runtime import RouteKVEngine

engine = RouteKVEngine.from_pretrained(
    base_model="meta-llama/Llama-3.2-1B",
    cost_model_path="models/cost_model.pt",
    tier_config_path="configs/llama_1b_tier3.yaml",
)

prompt = "Explain why KV cache movement dominates long-context inference."
out = engine.generate(prompt, max_new_tokens=256)
print(out)
```

What this does conceptually:

- Loads a base LLM plus the learned cost model.
- Builds a tiering plan per request / layer.
- Routes KV blocks between HBM, DRAM and the transformed store according to predicted cost.

### Example 2 — Integrating with an existing serving stack

You can treat RouteKV as a KV-tiering backend behind your existing sampler / scheduler:

```python
# conceptual example API; subject to change
from routekv.runtime import RouteKVSession
from my_serving_stack import RequestBatch

rk_session = RouteKVSession.load(
    model_name="meta-llama/Llama-3.2-1B",
    cost_model_path="models/cost_model.pt"
)

while True:
    batch: RequestBatch = scheduler.next_batch()

    # RouteKV decides which KV blocks stay on GPU vs CPU vs transformed store.
    logits, new_state = rk_session.step(
        input_ids=batch.input_ids,
        kv_state=batch.kv_state,
        metadata=batch.metadata,  # e.g., conversation id, route, workload hints
    )

    scheduler.return_step_results(batch.request_ids, logits, new_state)
```

This loop lets your existing scheduler own batching and admission control, while RouteKV owns KV placement and movement decisions under the hood.

### Example 3 — Experimenting with tiering policies

```bash
# Baseline: no RouteKV (e.g., vanilla vLLM LRU)
python experiments/benchmark_tiering.py \
  --config configs/llama_1b_baseline.yaml

# Learned cost model with 3-tier hierarchy
python experiments/benchmark_tiering.py \
  --config configs/llama_1b_tier3.yaml

# Aggressive compression for ultra-long contexts
python experiments/benchmark_tiering.py \
  --config configs/llama_1b_tier3_aggressive.yaml
```

Compare tokens/sec, GPU memory usage, and quality metrics across these configs to understand how tiering affects your workload.

---

## 📊 Research Goals

| Metric                 | Baseline (LRU-style) | RouteKV Target |
|------------------------|----------------------|----------------|
| Tokens/sec (long ctx)  | 1×                   | 2–4×           |
| GPU memory reduction   | 1×                   | 3–5×           |
| Quality (LongBench)    | 100%                 | ≥97%           |
| Cold-tier accuracy     | N/A                  | <2% drop       |

These are stretch goals to guide design, not guaranteed benchmarks.

---

## 📚 References

RouteKV is informed by recent work on KV cache compression, eviction, and routing, including (non-exhaustive):

- KV cache optimization and offloading for long-context LLMs.
- ScoutAttention-style layer-ahead prefetching.
- Transform-coding approaches for KV compression (e.g., KVTC-like methods).
- Layer-wise eviction policies (e.g., LAVa-like approaches).
- Importance/diversity-aware KV selection (e.g., MixKV-style scoring).
- Prefix-aware and route-aware KV reuse schemes.

For details, see the papers cited in the project’s `docs/` and notebooks.

---

## 🛠️ Requirements

- Python ≥ 3.10  
- PyTorch ≥ 2.2  
- CUDA ≥ 11.8 (tested on common datacenter GPUs)  
- Triton ≥ 2.1  
- `transformers`, `accelerate`, `vllm` (optional, for certain integrations)

Install all extras for development and experiments with:

```bash
pip install -e ".[dev]"
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE).
