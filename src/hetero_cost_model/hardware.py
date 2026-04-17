"""Hardware feature vectors for heterogeneous GPU targets.

The cost model consumes hardware as a continuous, normalized feature vector
so it can generalize across GPU architectures (V100→A100→H100→H200→B200)
instead of relying on discrete one-hot encodings. Adding a new device is a
one-line registry entry.

Feature vector semantics (5-dim, index → physical quantity):

  0: fp16_tflops    — peak FP16 tensor core throughput (TFLOPS)
  1: memory_gb      — HBM capacity (GB)
  2: bandwidth_gbs  — HBM bandwidth (GB/s); dominant predictor for memory-bound ops
  3: l2_cache_mb    — L2 cache size (MB); on-chip bandwidth amplifier. Chosen
                      over PCIe/NVLink bandwidth (NeuSight ASPLOS'25 convention):
                      interconnect bandwidth is irrelevant during a single-GPU
                      timed forward pass, so including it would inject
                      confounded signal.
  4: sm_count       — number of streaming multiprocessors (parallelism)

Deliberately **no architecture-generation ordinal**: an earlier draft included
``arch_gen`` ∈ {0.0 Volta, 0.33 Ampere, 0.67 Hopper, 1.0 Blackwell}, but with
only 3 training anchors this behaves as a disguised device-ID lookup. Removing
it forces the model to rely on genuine on-chip features, aligning with
NeuSight's "on-chip features only" convention and keeping the spec-only
zero-shot claim honest.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


HARDWARE_FEATURE_DIM: int = 5

# Normalization bounds — chosen to cover Blackwell with headroom,
# so all features land in roughly [0, 1].
_NORMALIZATION = (5000.0, 200.0, 9000.0, 100.0, 200.0)


@dataclass(frozen=True)
class Hardware:
    """Compact spec for a GPU execution target."""

    name: str
    fp16_tflops: float   # peak FP16 tensor TFLOPS
    memory_gb: float     # HBM capacity (GB)
    bandwidth_gbs: float # HBM bandwidth (GB/s)
    l2_cache_mb: float   # L2 cache size (MB); on-chip bandwidth amplifier
    sm_count: int        # number of streaming multiprocessors

    def to_vector(self, normalize: bool = True) -> List[float]:
        raw = [
            self.fp16_tflops,
            self.memory_gb,
            self.bandwidth_gbs,
            self.l2_cache_mb,
            float(self.sm_count),
        ]
        if not normalize:
            return raw
        return [a / b for a, b in zip(raw, _NORMALIZATION)]


HARDWARE_REGISTRY: Dict[str, Hardware] = {
    # All values from NVIDIA spec sheets, FP16 tensor-core TFLOPS without
    # 2:4 structured sparsity (dense), HBM/GDDR capacity, peak memory
    # bandwidth, L2 cache size, streaming-multiprocessor count.
    # Adding a new device is a one-line registry entry + one-line add to
    # scripts/modal_profiling.py (if on Modal).
    # --- Training anchors (6 points spanning 5 architecture generations) ---
    "v100": Hardware("V100", fp16_tflops=125,  memory_gb=32,  bandwidth_gbs=900,  l2_cache_mb=6,  sm_count=80),   # Volta     (sm_70)
    "t4":   Hardware("T4",   fp16_tflops=65,   memory_gb=16,  bandwidth_gbs=320,  l2_cache_mb=4,  sm_count=40),   # Turing    (sm_75)
    "a100": Hardware("A100", fp16_tflops=312,  memory_gb=40,  bandwidth_gbs=1555, l2_cache_mb=40, sm_count=108),  # Ampere    (sm_80, GA100)
    # A10 is shorthand — Modal schedules us onto A10G (AWS G5 variant) which
    # has 80 SMs / ~70 TFLOPS dense FP16 tensor (vs datacenter A10's 72 / 125).
    # Values below reflect A10G since that's what we actually measure.
    "a10":  Hardware("A10",  fp16_tflops=70,   memory_gb=24,  bandwidth_gbs=600,  l2_cache_mb=6,  sm_count=80),   # Ampere    (sm_86, GA102, A10G)
    "l4":   Hardware("L4",   fp16_tflops=121,  memory_gb=24,  bandwidth_gbs=300,  l2_cache_mb=48, sm_count=58),   # Ada       (sm_89)
    "h100": Hardware("H100", fp16_tflops=1979, memory_gb=80,  bandwidth_gbs=3350, l2_cache_mb=50, sm_count=132),  # Hopper    (sm_90)
    # --- Few-shot test target ---
    "h200": Hardware("H200", fp16_tflops=1979, memory_gb=141, bandwidth_gbs=4800, l2_cache_mb=50, sm_count=132),  # Hopper+   (sm_90)
    # --- Hero zero-shot target ---
    "b200": Hardware("B200", fp16_tflops=4500, memory_gb=180, bandwidth_gbs=8000, l2_cache_mb=96, sm_count=160),  # Blackwell (sm_100)
}


__all__ = ["Hardware", "HARDWARE_REGISTRY", "HARDWARE_FEATURE_DIM"]
