"""Analytical baseline: GPU Roofline model.

Estimates latency as max(compute-bound, memory-bound) per op, summed over
all ops in the graph. Used as a non-learned reference to show that the GNN
cost model learns something beyond simple peak-performance formulas.
"""
from __future__ import annotations

from hetero_cost_model.graph import GraphRepr
from hetero_cost_model.hardware import Hardware
from hetero_cost_model.strategies import InferenceConfig


def roofline_latency(
    graph: GraphRepr,
    config: InferenceConfig,
    hardware: Hardware,
) -> float:
    """GPU roofline latency estimate in milliseconds.

    Each op is bounded by either compute (FLOPs / peak_TFLOPS) or memory
    bandwidth (bytes / peak_BW). FLOPs are scaled by batch_size since the
    graph stores per-token/per-sample estimates.
    """
    peak_flops = hardware.fp16_tflops * 1e12   # ops/s
    peak_bw    = hardware.bandwidth_gbs * 1e9  # bytes/s

    total_ms = 0.0
    for node in graph.nodes:
        scaled_flops = node.flops * config.batch_size
        compute_ms  = scaled_flops / peak_flops * 1000.0 if peak_flops else 0.0
        memory_ms   = node.memory_bytes / peak_bw * 1000.0 if peak_bw else 0.0
        total_ms += max(compute_ms, memory_ms)
    return total_ms


__all__ = ["roofline_latency"]
