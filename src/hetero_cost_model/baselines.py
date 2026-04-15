"""Analytical & trivial baselines — non-learned references for the cost model.

- ``roofline_latency``: peak-FLOPS vs. peak-bandwidth bound per op, plus a
  PCIe transfer penalty at every CPU↔GPU boundary. ~50 lines, pure formula.
- ``random_strategy_selector``: a lower bound on ranking quality.
"""
from __future__ import annotations

import random
from typing import List, Optional, Tuple

from hetero_cost_model.graph import GraphRepr
from hetero_cost_model.hardware import Hardware
from hetero_cost_model.strategies import GPU, Strategy


# Rough single-socket CPU approximations; refined at profiling time.
_DEFAULT_CPU_TFLOPS = 1.0
_DEFAULT_CPU_BW_GBS = 100.0


def roofline_latency(
    graph: GraphRepr,
    strategy: Strategy,
    hardware: Hardware,
    *,
    cpu_tflops: float = _DEFAULT_CPU_TFLOPS,
    cpu_bw_gbs: float = _DEFAULT_CPU_BW_GBS,
    transfer_bytes_per_boundary: float = 1e6,
) -> float:
    """Peak-bound latency estimate (ms) with a PCIe transfer penalty."""
    total_ms = 0.0
    prev_device: Optional[int] = None
    for node, device in zip(graph.nodes, strategy.placements):
        peak_flops, peak_bw = _peak_perf(device, hardware, cpu_tflops, cpu_bw_gbs)
        compute_ms = node.flops / peak_flops * 1000.0 if peak_flops else 0.0
        memory_ms = node.memory_bytes / peak_bw * 1000.0 if peak_bw else 0.0
        total_ms += max(compute_ms, memory_ms)
        if prev_device is not None and prev_device != device:
            total_ms += transfer_bytes_per_boundary / (hardware.pcie_gbs * 1e9) * 1000.0
        prev_device = device
    return total_ms


def random_strategy_selector(strategies: List[Strategy], seed: int = 0) -> Strategy:
    return random.Random(seed).choice(strategies)


def _peak_perf(
    device: int, hw: Hardware, cpu_tflops: float, cpu_bw_gbs: float,
) -> Tuple[float, float]:
    if device == GPU:
        return hw.fp16_tflops * 1e12, hw.bandwidth_gbs * 1e9
    return cpu_tflops * 1e12, cpu_bw_gbs * 1e9


__all__ = ["roofline_latency", "random_strategy_selector"]
