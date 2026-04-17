"""Hardware feature vectors for heterogeneous GPU targets.

The cost model consumes hardware as a continuous, normalized feature vector
so it can generalize across GPU architectures (V100→A100→H100→H200→B200)
instead of relying on discrete one-hot encodings. Adding a new device is a
one-line registry entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


HARDWARE_FEATURE_DIM: int = 5

# Normalization bounds — chosen to cover Blackwell with headroom,
# so all features land in roughly [0, 1].
_NORMALIZATION = (5000.0, 200.0, 9000.0, 128.0, 200.0)


@dataclass(frozen=True)
class Hardware:
    """Compact spec for a GPU execution target."""

    name: str
    fp16_tflops: float   # peak FP16 tensor TFLOPS
    memory_gb: float     # HBM capacity (GB)
    bandwidth_gbs: float # HBM bandwidth (GB/s)
    pcie_gbs: float      # host↔device interconnect bandwidth (GB/s)
    sm_count: int        # number of streaming multiprocessors

    def to_vector(self, normalize: bool = True) -> List[float]:
        raw = [
            self.fp16_tflops,
            self.memory_gb,
            self.bandwidth_gbs,
            self.pcie_gbs,
            float(self.sm_count),
        ]
        if not normalize:
            return raw
        return [a / b for a, b in zip(raw, _NORMALIZATION)]


HARDWARE_REGISTRY: Dict[str, Hardware] = {
    # --- Training GPUs (PSC) ---
    "v100": Hardware("V100", fp16_tflops=125,  memory_gb=32,  bandwidth_gbs=900,  pcie_gbs=16,  sm_count=80),
    "h100": Hardware("H100", fp16_tflops=1979, memory_gb=80,  bandwidth_gbs=3350, pcie_gbs=64,  sm_count=132),
    # --- Training GPU (Modal) ---
    "a100": Hardware("A100", fp16_tflops=312,  memory_gb=40,  bandwidth_gbs=1555, pcie_gbs=64,  sm_count=108),
    # --- Zero-shot test GPUs (Modal) ---
    "h200": Hardware("H200", fp16_tflops=1979, memory_gb=141, bandwidth_gbs=4800, pcie_gbs=64,  sm_count=132),
    "b200": Hardware("B200", fp16_tflops=4500, memory_gb=180, bandwidth_gbs=8000, pcie_gbs=128, sm_count=160),
}


__all__ = ["Hardware", "HARDWARE_REGISTRY", "HARDWARE_FEATURE_DIM"]
