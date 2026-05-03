"""
RouteKV Compiler
================
A route-aware, context-sensitive KV cache memory tiering compiler
for long-context LLM inference.

Atoms:
  - Placement  : which tier holds each KV block
  - Representation : encoding per tier (FP16 / INT4 / KVTC)
  - Selection  : importance + diversity scoring
  - Scheduling : async prefetch / recall timing
  - Sharing    : cross-request prefix/inter-task reuse
"""

__version__ = "0.1.0"
__author__ = "Vamsi Sai Ranga Mangina"

from routekv.compiler.tier_plan import TierPlan
from routekv.runtime.engine import RouteKVEngine

__all__ = ["TierPlan", "RouteKVEngine"]
