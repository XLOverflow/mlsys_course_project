"""Wall-clock profiling primitives.

GPU-aware timing harness. Two timing paths:

  - **CUDA Event timing** (preferred when ``device`` starts with ``cuda``):
    Uses ``torch.cuda.Event(enable_timing=True)`` pairs. Measures GPU-side
    elapsed time, decoupling from CPU / Python dispatch jitter which matters
    on shared-host nodes (PSC) and for small kernels (< 5 ms).
  - **Wall clock** (``time.perf_counter``) for CPU paths or when CUDA is
    unavailable. Still synchronizes CUDA before/after if a CUDA device is
    present, so the number is at least a valid upper bound.

Defaults changed 2026-04-17 (team review):
  warmup  10 → 50   (more headroom for cuDNN autotune on small kernels)
  runs    50 → 100  (tightens p50 estimate; runtime cost ~2×)

Training target is ``p50_ms`` rather than ``mean_ms`` — shared-host long-tail
does not bias the median. ``mean_ms`` is kept for noise diagnostics.
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

    @property
    def cov(self) -> float:
        """Coefficient of variation. Used for the ``noisy`` quality flag."""
        return self.std_ms / self.mean_ms if self.mean_ms else 0.0

    @property
    def noisy(self) -> bool:
        """``True`` when run-to-run variation exceeds 5 %."""
        return self.cov > 0.05

    def as_dict(self) -> dict:
        return {
            "mean_ms": self.mean_ms,
            "std_ms": self.std_ms,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "n_runs": self.n_runs,
            "noisy": self.noisy,
        }


def profile_callable(
    fn: Callable[[], None],
    *,
    warmup: int = 50,
    runs: int = 100,
    device: Optional[str] = None,
) -> ProfileResult:
    """Run ``fn`` repeatedly and return wall-clock statistics in milliseconds.

    Uses CUDA events when ``device`` is a CUDA device (GPU-side timing,
    insensitive to CPU jitter); falls back to ``time.perf_counter`` otherwise.
    """
    use_cuda_events = (
        device is not None
        and device.startswith("cuda")
        and torch.cuda.is_available()
    )

    for _ in range(warmup):
        fn()
    _sync(device)

    if use_cuda_events:
        samples = _measure_cuda_events(fn, runs)
    else:
        samples = _measure_perf_counter(fn, runs, device)

    return _summarize(samples)


# --- Measurement paths -------------------------------------------------------

def _measure_cuda_events(fn: Callable[[], None], runs: int) -> List[float]:
    """GPU-side timing via paired events; isolates CPU dispatch jitter."""
    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(runs)]

    for i in range(runs):
        start_events[i].record()
        fn()
        end_events[i].record()

    # Single sync at the end; events queue on the stream without blocking.
    torch.cuda.synchronize()

    return [start_events[i].elapsed_time(end_events[i]) for i in range(runs)]


def _measure_perf_counter(
    fn: Callable[[], None], runs: int, device: Optional[str],
) -> List[float]:
    """CPU wall-clock timing. Still syncs CUDA if present (conservative)."""
    samples: List[float] = []
    for _ in range(runs):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return samples


def _sync(device: Optional[str]) -> None:
    if device and device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


# --- Summary -----------------------------------------------------------------

def _summarize(samples: List[float]) -> ProfileResult:
    samples = sorted(samples)
    n = len(samples)
    mean = sum(samples) / n
    std = (sum((s - mean) ** 2 for s in samples) / n) ** 0.5
    p50 = samples[n // 2]
    p95 = samples[min(int(n * 0.95), n - 1)]
    return ProfileResult(mean, std, p50, p95, n)


def time_decode_step(
    model,
    input_ids,
    *,
    extra: Optional[dict] = None,
    warmup: int = 50,
    runs: int = 100,
    device: Optional[str] = None,
) -> ProfileResult:
    """Profile a single autoregressive decode step (one-token forward + KV cache).

    The protocol mirrors steady-state autoregressive decoding inside HF
    ``generate()``: run prefill once over ``input_ids`` to warm the KV
    cache, then time repeated single-token forwards that *consume* the
    cache without growing it. Each timed call simulates "produce token
    i+1 given a cache of size i" at a fixed cache size, which is what
    serving-side cost models actually consume.

    Only applies to decoder-only and encoder-decoder models. Encoder-only
    models (BERT family) have no autoregressive decode and are rejected
    by the caller, not here.

    Parameters
    ----------
    model
        A loaded HuggingFace model with ``use_cache=True`` support.
    input_ids
        Prompt tokens [batch, seq_len]. Used once for prefill to populate
        the KV cache.
    extra
        Optional kwargs passed both to the prefill call and to each
        decode-step call (e.g. ``decoder_input_ids`` for encoder-decoder
        models is set automatically inside this function based on the
        prefill output).

    Returns
    -------
    :class:`ProfileResult` summarizing per-token decode latency. The
    measurement excludes prefill cost.
    """
    extra = dict(extra or {})

    # --- Prefill once to populate KV cache --------------------------------
    with torch.inference_mode():
        out = model(input_ids, use_cache=True, **extra)
    past = getattr(out, "past_key_values", None)
    if past is None:
        raise RuntimeError(
            "model returned no past_key_values; this model probably does "
            "not support KV-cache decoding (encoder-only?)."
        )

    # Last token in the prompt becomes the "new token" the decode step
    # nominally produced — for fixed-cache-size timing it doesn't matter
    # what value we pick, only that the shape is [batch, 1].
    one_token = input_ids[:, -1:].clone()
    decode_kwargs = {"use_cache": True, "past_key_values": past}

    # For encoder-decoder, the encoder hidden states are baked into past
    # already (T5 caches encoder cross-attention KV inside past_key_values),
    # so we drop the encoder-side `decoder_input_ids` from `extra` here.
    decode_extra = {k: v for k, v in extra.items() if k != "decoder_input_ids"}

    def decode_step() -> None:
        with torch.inference_mode():
            model(one_token, **decode_kwargs, **decode_extra)

    return profile_callable(decode_step, warmup=warmup, runs=runs, device=device)


__all__ = ["ProfileResult", "profile_callable", "time_decode_step"]
