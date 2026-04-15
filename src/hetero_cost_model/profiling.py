"""Wall-clock profiling primitives.

Focused on a clean, GPU-aware timing harness (warmup + repeated measurement
with CUDA synchronization). Actually mapping a :class:`Strategy` onto a
compiled fx GraphModule with per-op device placement is the job of the
Week-1 Day-3 profiling script that runs on the shared GPUs.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch


@dataclass
class ProfileResult:
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    n_runs: int

    def as_dict(self) -> dict:
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "n_runs": self.n_runs,
        }


def profile_callable(
    fn: Callable[[], None],
    *,
    warmup: int = 10,
    runs: int = 50,
    device: Optional[str] = None,
) -> ProfileResult:
    """Run ``fn`` repeatedly and return wall-clock statistics in milliseconds."""
    for _ in range(warmup):
        fn()
    _sync(device)

    samples: List[float] = []
    for _ in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)

    return _summarize(samples)


def _sync(device: Optional[str]) -> None:
    if device and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _summarize(samples: List[float]) -> ProfileResult:
    samples.sort()
    n = len(samples)
    mean = sum(samples) / n
    std = (sum((s - mean) ** 2 for s in samples) / n) ** 0.5
    p50 = samples[n // 2]
    p95 = samples[min(int(n * 0.95), n - 1)]
    return ProfileResult(mean, std, p50, p95, n)


__all__ = ["ProfileResult", "profile_callable"]
