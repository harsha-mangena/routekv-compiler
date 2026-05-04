"""
DecodeStepTrace — schema for one profiled decode step.

Fields
------
step                : decode step index
sequence_length     : total tokens in context at this step
layer               : transformer layer index being profiled
attention_density   : fraction of attention heads with non-negligible scores (sparsity proxy)
kv_bytes_touched    : bytes of KV data read/written in this step
gpu_stall_us        : estimated GPU stall time in microseconds (HBM bandwidth bound)
cpu_stall_us        : estimated CPU stall time in microseconds (PCIe/DRAM bound)
context_intensity   : categorical workload type: 'retrieval' | 'summarisation' | 'chat' | 'reasoning'
"""
from typing import Literal
from pydantic import BaseModel, Field


ContextType = Literal["retrieval", "summarisation", "chat", "reasoning"]


class DecodeStepTrace(BaseModel):
    step: int
    sequence_length: int
    layer: int
    attention_density: float = Field(ge=0.0, le=1.0)
    kv_bytes_touched: int
    gpu_stall_us: float
    cpu_stall_us: float
    context_intensity: ContextType = "chat"
