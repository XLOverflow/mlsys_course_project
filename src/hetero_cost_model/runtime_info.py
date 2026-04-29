"""Runtime GPU introspection for CSV provenance columns.

``actual_gpu_name`` / ``actual_mem_gb`` / ``actual_sm_count`` guard against
Modal silently upgrading SKUs (``gpu="A100"`` may land on A100-80GB; ``H100``
on B200). When loading the training CSV we look hardware up by
``actual_gpu_name`` rather than the declared ``gpu`` label.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass(frozen=True)
class RuntimeGPUInfo:
    actual_gpu_name: str    # e.g. "NVIDIA A100-SXM4-40GB"
    actual_mem_gb: float    # from torch.cuda.get_device_properties().total_memory
    actual_sm_count: int    # streaming multiprocessor count
    registry_key: str       # canonical key: v100/a10/a100/b200/h100/l4, "" if unknown
    cuda_version: str       # torch.version.cuda (may be "" on CPU)
    driver_version: str     # nvidia-smi driver version (best-effort)
    torch_version: str

    def as_dict(self) -> dict:
        return {
            "actual_gpu_name": self.actual_gpu_name,
            "actual_mem_gb": self.actual_mem_gb,
            "actual_sm_count": self.actual_sm_count,
            "cuda_version": self.cuda_version,
            "driver_version": self.driver_version,
            "torch_version": self.torch_version,
        }


def current_gpu_info() -> RuntimeGPUInfo:
    """Query the current CUDA device, or return a CPU placeholder."""
    if not torch.cuda.is_available():
        return RuntimeGPUInfo(
            actual_gpu_name="cpu",
            actual_mem_gb=0.0,
            actual_sm_count=0,
            registry_key="",
            cuda_version="",
            driver_version="",
            torch_version=torch.__version__,
        )

    props = torch.cuda.get_device_properties(0)
    name = torch.cuda.get_device_name(0)
    return RuntimeGPUInfo(
        actual_gpu_name=name,
        actual_mem_gb=props.total_memory / 1e9,
        actual_sm_count=props.multi_processor_count,
        registry_key=gpu_name_to_registry_key(name),
        cuda_version=torch.version.cuda or "",
        driver_version=_driver_version(),
        torch_version=torch.__version__,
    )


# --- name → registry key -----------------------------------------------------

# Patterns are checked top-to-bottom, so more-specific substrings come first.
# A10G is Modal/AWS's variant of A10 (different SM count + slightly different
# clocks); we map it to the "a10" registry entry whose spec we've updated to
# reflect A10G since that's what Modal actually schedules.
_REGISTRY_PATTERNS = [
    (re.compile(r"\bB200\b",   re.IGNORECASE), "b200"),
    (re.compile(r"\bH100\b",   re.IGNORECASE), "h100"),
    (re.compile(r"\bA100\b",   re.IGNORECASE), "a100"),
    (re.compile(r"\bA10G?\b",  re.IGNORECASE), "a10"),   # matches A10 and A10G
    (re.compile(r"\bL4\b",     re.IGNORECASE), "l4"),
    (re.compile(r"\bV100\b",   re.IGNORECASE), "v100"),  # registered but unused
]


def gpu_name_to_registry_key(device_name: str) -> str:
    """Map ``torch.cuda.get_device_name()`` to our ``HARDWARE_REGISTRY`` key.

    Returns ``""`` if the device isn't one of V100/A10/A100/B200/H100/L4.
    Callers should treat empty string as "data row must be discarded" —
    it signals Modal gave us an unexpected SKU or we're on a dev GPU.
    """
    for pattern, key in _REGISTRY_PATTERNS:
        if pattern.search(device_name):
            return key
    return ""


# --- nvidia-smi driver version (best effort) ---------------------------------

def _driver_version() -> str:
    import subprocess
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            timeout=3,
        )
        return out.decode().strip().split("\n")[0]
    except (OSError, subprocess.SubprocessError):
        return ""


__all__ = ["RuntimeGPUInfo", "current_gpu_info", "gpu_name_to_registry_key"]
