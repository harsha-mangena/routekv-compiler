# 🧠 RouteKV Compiler

> **A route-aware, context-sensitive KV cache memory tiering compiler for long-context LLM inference.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![CUDA](https://img.shields.io/badge/CUDA-11.8%2B-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Research](https://img.shields.io/badge/Status-Research%20Prototype-orange)]()

---

## 🔬 Motivation

LLM inference is memory-bound at low concurrency — the dominant bottleneck is **moving KV tensors between HBM, CPU DRAM, and the attention cores**, not raw compute. Current systems treat this as an afterthought:

- **vLLM** pages KV blocks with LRU eviction — workload-agnostic [web:60][web:61]
- **SGLang RadixAttention** handles prefix sharing but not per-layer tier decisions [web:65][web:68]
- **ScoutAttention** hides CPU latency with layer-ahead prefetch but uses fixed heuristics [web:81]
- **KVTC / LAVa / MixKV** compress or evict at a single stage, ignoring downstream routing [web:45][web:69][web:77]

**The gap:** No system jointly decides *where* KV lives, *how* it is encoded, *when* to move it, and *which requests share it* — all together, at runtime, driven by workload context.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RouteKV Compiler Stack                        │
│                                                                  │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐ │
│  │  Profiler    │──▶│ Cost Model   │──▶│  Tier Plan Compiler  │ │
│  │  (offline)   │   │  (learned)   │   │  (online/JIT)        │ │
│  └──────────────┘   └──────────────┘   └──────────────────────┘ │
│          │                 │                      │              │
│          ▼                 ▼                      ▼              │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              RouteKV Runtime Engine                        │  │
│  │                                                            │  │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │  │
│  │  │  HBM    │  │ CPU DRAM │  │ Transform│  │  Shared  │  │  │
│  │  │  Pool   │  │  Pool    │  │ Coded    │  │  Store   │  │  │
│  │  │(hot)    │  │(warm)    │  │(cold)    │  │(reuse)   │  │  │
│  │  └─────────┘  └──────────┘  └──────────┘  └──────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│          │                                                       │
│          ▼                                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Attention Kernel Dispatcher (CUDA + Triton)              │  │
│  │  - Tier-aware attention (HBM-only / GPU-CPU co-attention) │  │
│  │  - Block-wise sparse scoring                              │  │
│  │  - Asynchronous PCIe transfer streams                     │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 Five Atoms (Atom of Thoughts)

| Atom | What it solves | Key insight from literature |
|------|---------------|-----------------------------|
| **Placement** | Where does each KV block live? | Tiered: HBM → DRAM → compressed store |
| **Representation** | How is each tier encoded? | KVTC transform coding for cold; INT4 for warm; FP16 for hot |
| **Selection** | Which blocks to promote/evict? | LAVa head-diversity + route-criticality score |
| **Scheduling** | When to move data? | ScoutAttention layer-ahead async prefetch |
| **Sharing** | Cross-request KV reuse? | RadixAttention prefix trie + similarity-based inter-task reuse |

---

## 🌲 Tree of Thoughts Expansion

```
RouteKV Problem Root
│
├── Branch A: Compression-first
│   ├── KVTC (transform coding, 20x compression) ✅ adopted
│   ├── LAVa (layer-wise dynamic budget) ✅ adopted
│   └── MixKV (importance + diversity) ✅ adopted as scoring
│
├── Branch B: Offload-first
│   ├── ScoutAttention (layer-ahead prefetch) ✅ adopted
│   ├── llm-d (filesystem-backed KV) ✅ as cold tier
│   └── GPU-CPU co-attention ✅ as runtime dispatch
│
├── Branch C: Routing-first [NOVEL]
│   ├── Physics of KV compression (route criticality)
│   ├── Workload-type classification → tier policy
│   └── Cost model: predict tier assignment per block
│
└── Branch D: Sharing-first
    ├── RadixAttention prefix trie
    ├── Inter-task similarity reuse (NeurIPS 2025)
    └── Distributed KV store across replicas
```

**RouteKV's innovation**: Branch C is the unimplemented connector — a *learned cost model* that takes workload context + hardware state and outputs a per-block tier plan, replacing all hand-tuned heuristics.

---

## 📦 Repository Structure

```
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

```bash
git clone https://github.com/harsha-mangena/routekv-compiler
cd routekv-compiler
pip install -e ".[dev]"

# Run profiler on a small model
python -m routekv.profiler.collect \
  --model meta-llama/Llama-3.2-1B \
  --dataset longbench \
  --output traces/llama_1b_traces.pkl

# Train cost model
python -m routekv.cost_model.train \
  --traces traces/llama_1b_traces.pkl \
  --output models/cost_model.pt

# Run benchmark
python experiments/benchmark_tiering.py \
  --config configs/llama_1b_tier3.yaml
```

---

## 📊 Research Goals

| Metric | Baseline (vLLM LRU) | RouteKV Target |
|--------|--------------------|-----------------|
| Tokens/sec (long ctx) | 1x | 2–4x |
| GPU memory reduction | 1x | 3–5x |
| Quality (LongBench) | 100% | ≥97% |
| Cold-tier accuracy | N/A | <2% accuracy drop |

---

## 📚 Key References

- **KV Cache Optimization Survey** (arXiv 2603.20397, 2026)
- **KV Cache Offloading for Context-Intensive Tasks** (arXiv 2604.08426, 2026)
- **ScoutAttention** (arXiv 2603.27138, 2026)
- **KVTC: KV Cache Transform Coding** (arXiv 2511.01815, 2025)
- **LAVa: Layer-wise KV Cache Eviction** (arXiv 2509.09754, 2025)
- **MixKV: Importance + Diversity** (ICLR 2026, arXiv 2510.20707)
- **Physics of KV Compression** (arXiv 2603.01426, 2026)
- **End-to-End Transformer PIM Acceleration** (arXiv 2601.14260, 2025)
- **G-KV: Global Attention Eviction** (arXiv 2512.00504, 2025)
- **PagedEviction** (arXiv 2509.04377, 2025)

---

## 🛠️ Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.2
- CUDA ≥ 11.8 (or Google Colab T4/A100)
- Triton ≥ 2.1
- transformers, accelerate, vllm (optional)

---

## 📄 License

MIT License. See [LICENSE](LICENSE).
