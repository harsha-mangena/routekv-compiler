"""
RouteKV Compiler IR — core data structures for KV page scheduling.

Atom map
--------
A KVPage is the atomic unit of memory management in RouteKV.
Each page carries:
  - spatial identity  (layer, head_group, token range)
  - size estimate     (bytes)
  - two routing scores that drive tier placement:
      hotness      — recency / attention density proxy
      route_score  — route-criticality: how much answer routes pass through this page

A PlacementDecision is the output atom from the compiler planner:
  which tier (hbm | dram | storage), which encoding, and whether to prefetch.
"""
from typing import Literal
from pydantic import BaseModel, Field


class KVPage(BaseModel):
    """Atomic KV cache page."""
    layer: int = Field(ge=0, description="Transformer layer index")
    head_group: int = Field(ge=0, description="Grouped-query attention head group index")
    start_token: int = Field(ge=0, description="First token position in this page")
    end_token: int = Field(ge=0, description="Last token position in this page (exclusive)")
    bytes_estimate: int = Field(gt=0, description="Estimated memory footprint in bytes")
    hotness: float = Field(default=0.0, ge=0.0, le=1.0,
                           description="Normalised recency/attention-density proxy [0,1]")
    route_score: float = Field(default=0.0, ge=0.0, le=1.0,
                               description="Route-criticality score: fraction of answer routes passing through this page [0,1]")

    @property
    def page_id(self) -> str:
        return f"L{self.layer}:G{self.head_group}:{self.start_token}-{self.end_token}"

    @property
    def composite_score(self) -> float:
        """Weighted composite placement score."""
        return 0.6 * self.route_score + 0.4 * self.hotness


Tier = Literal["hbm", "dram", "storage"]
Encoding = Literal["fp16", "int8", "int4", "transform", "landmark"]


class PlacementDecision(BaseModel):
    """Compiler output: where and how to store a KV page."""
    page_id: str
    tier: Tier = Field(description="Memory tier for this page")
    encoding: Encoding = Field(description="Storage encoding/compression format")
    prefetch: bool = Field(default=False, description="Whether to proactively prefetch this page before it is needed")
    evict_after_step: int = Field(default=-1, description="Decode step after which this page can be evicted (-1 = never)")
