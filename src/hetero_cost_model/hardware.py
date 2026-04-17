"""Hardware feature vectors for heterogeneous GPU targets.

The cost model consumes hardware as a continuous, normalized feature vector
so it can generalize across GPU architectures (V100→A100→H100→H200→B200)
instead of relying on discrete one-hot encodings. Adding a new device is a
one-line registry entry.

Feature vector semantics (index → physical quantity):

  0: fp16_tflops    — peak FP16 tensor core throughput (TFLOPS)
  1: memory_gb      — HBM capacity (GB)
  2: bandwidth_gbs  — HBM bandwidth (GB/s); dominant predictor for memory-bound ops
  3: l2_cache_mb    — L2 cache size (MB); on-chip feature that affects bandwidth
                      amplification and small-kernel latency. Chosen over PCIe
                      bandwidth (NeuSight ASPLOS'25 convention): interconnect
                      bandwidth is irrelevant during a single-GPU timed forward
                      pass, so including it would inject confounded signal.
  4: sm_count       — number of streaming multiprocessors (parallelism)
  5: arch_gen       — architecture generation as ordinal (Volta=0.0, Ampere=0.33,
                      Hopper=0.67, Blackwell=1.0). Effectively acts as a
                      smoothed generation index to help extrapolation to
                      Blackwell; may get scrutinized in ablation for being a
                      disguised discrete lookup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


HARDWARE_FEATURE_DIM: int = 6

# Normalization bounds — chosen to cover Blackwell with headroom,
# so all features land in roughly [0, 1].
# Last dimension (arch_gen) is already in [0, 1] (pre-normalized ordinal).
_NORMALIZATION = (5000.0, 200.0, 9000.0, 100.0, 200.0, 1.0)


@dataclass(frozen=True)
class Hardware:
    """Compact spec for a GPU execution target."""

    name: str
    fp16_tflops: float   # peak FP16 tensor TFLOPS
    memory_gb: float     # HBM capacity (GB)
    bandwidth_gbs: float # HBM bandwidth (GB/s)
    l2_cache_mb: float   # L2 cache size (MB); on-chip bandwidth amplifier
    sm_count: int        # number of streaming multiprocessors
    arch_gen: float      # GPU architecture generation (Volta=0.0 → Blackwell=1.0)

    def to_vector(self, normalize: bool = True) -> List[float]:
        raw = [
            self.fp16_tflops,
            self.memory_gb,
            self.bandwidth_gbs,
            self.l2_cache_mb,
            float(self.sm_count),
            self.arch_gen,
        ]
        if not normalize:
            return raw
        return [a / b for a, b in zip(raw, _NORMALIZATION)]


HARDWARE_REGISTRY: Dict[str, Hardware] = {
    # L2 cache values from NVIDIA spec sheets:
    #   V100 (Volta):     6 MB    A100 (Ampere):  40 MB   H100 (Hopper):    50 MB
    #   H200 (Hopper):   50 MB    B200 (Blackwell): 96 MB (announcement; subject to final docs)
    # --- Training GPUs (PSC) ---
    "v100": Hardware("V100", fp16_tflops=125,  memory_gb=32,  bandwidth_gbs=900,  l2_cache_mb=6,  sm_count=80,  arch_gen=0.00),   # Volta
    "h100": Hardware("H100", fp16_tflops=1979, memory_gb=80,  bandwidth_gbs=3350, l2_cache_mb=50, sm_count=132, arch_gen=0.67),   # Hopper
    # --- Training GPU (Modal) ---
    "a100": Hardware("A100", fp16_tflops=312,  memory_gb=40,  bandwidth_gbs=1555, l2_cache_mb=40, sm_count=108, arch_gen=0.33),   # Ampere
    # --- Few-shot test GPU (Modal) ---
    "h200": Hardware("H200", fp16_tflops=1979, memory_gb=141, bandwidth_gbs=4800, l2_cache_mb=50, sm_count=132, arch_gen=0.67),   # Hopper
    # --- Zero-shot test GPU (Modal) ---
    "b200": Hardware("B200", fp16_tflops=4500, memory_gb=180, bandwidth_gbs=8000, l2_cache_mb=96, sm_count=160, arch_gen=1.00),   # Blackwell
}


__all__ = ["Hardware", "HARDWARE_REGISTRY", "HARDWARE_FEATURE_DIM"]
