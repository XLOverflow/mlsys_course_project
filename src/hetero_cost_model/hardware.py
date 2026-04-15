"""Hardware feature vectors for heterogeneous GPU/CPU targets.

The cost model consumes hardware as a continuous, normalized feature vector
so it can generalize across architectures (V100 / H100 / B200) instead of
relying on discrete one-hot encodings. Adding a new device is a one-line
registry entry.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


HARDWARE_FEATURE_DIM: int = 6

# Normalization constants — chosen to cover the current top-of-stack
# (Blackwell) with room to spare, so all features land in roughly [0, 1].
_NORMALIZATION = (5000.0, 200.0, 9000.0, 128.0, 128.0, 5.0)


@dataclass(frozen=True)
class Hardware:
    """Compact spec for a CPU/GPU execution target."""

    name: str
    fp16_tflops: float       # peak FP16 tensor TFLOPS
    memory_gb: float         # HBM capacity
    bandwidth_gbs: float     # HBM bandwidth
    pcie_gbs: float          # host<->device interconnect bandwidth
    cpu_cores: int
    cpu_freq_ghz: float

    def to_vector(self, normalize: bool = True) -> List[float]:
        raw = [
            self.fp16_tflops,
            self.memory_gb,
            self.bandwidth_gbs,
            self.pcie_gbs,
            float(self.cpu_cores),
            self.cpu_freq_ghz,
        ]
        if not normalize:
            return raw
        return [a / b for a, b in zip(raw, _NORMALIZATION)]


HARDWARE_REGISTRY: Dict[str, Hardware] = {
    "v100": Hardware("V100", fp16_tflops=125,  memory_gb=32,  bandwidth_gbs=900,  pcie_gbs=16,  cpu_cores=32, cpu_freq_ghz=2.5),
    "h100": Hardware("H100", fp16_tflops=1979, memory_gb=80,  bandwidth_gbs=3350, pcie_gbs=64,  cpu_cores=64, cpu_freq_ghz=3.0),
    "b200": Hardware("B200", fp16_tflops=4500, memory_gb=180, bandwidth_gbs=8000, pcie_gbs=128, cpu_cores=96, cpu_freq_ghz=3.5),
}


__all__ = ["Hardware", "HARDWARE_REGISTRY", "HARDWARE_FEATURE_DIM"]
