"""Regression tests for GPU-name → registry-key resolution.

These patterns are load-bearing: if they misresolve, profiling CSVs won't
merge into training samples. In particular the A10/A10G alias matters
because Modal schedules A10G even when the app requests ``gpu="A10"``.
"""
from hetero_cost_model.hardware import HARDWARE_REGISTRY
from hetero_cost_model.runtime_info import gpu_name_to_registry_key


def test_each_registry_entry_is_reachable_from_a_device_name():
    """Every HARDWARE_REGISTRY key must be reachable from some realistic
    ``torch.cuda.get_device_name()`` string."""
    device_names_for_key = {
        "v100": ["Tesla V100-SXM2-32GB", "Tesla V100-PCIE-32GB"],
        "t4":   ["Tesla T4"],
        "a100": ["NVIDIA A100-SXM4-40GB", "NVIDIA A100 80GB HBM3"],
        "a10":  ["NVIDIA A10", "NVIDIA A10G"],   # A10G is Modal's AWS variant
        "l4":   ["NVIDIA L4"],
        "h100": ["NVIDIA H100 80GB HBM3", "NVIDIA H100-SXM5"],
        "h200": ["NVIDIA H200", "NVIDIA H200 141GB HBM3e"],
        "b200": ["NVIDIA B200"],
    }
    for key in HARDWARE_REGISTRY:
        assert key in device_names_for_key, (
            f"HARDWARE_REGISTRY has '{key}' but no device-name coverage in tests"
        )
        for name in device_names_for_key[key]:
            assert gpu_name_to_registry_key(name) == key, (
                f"'{name}' did not resolve to '{key}'"
            )


def test_a10g_resolves_to_a10():
    """Explicit test for the Modal SKU quirk: requesting gpu='A10' gives A10G."""
    assert gpu_name_to_registry_key("NVIDIA A10G") == "a10"


def test_unknown_gpu_returns_empty_string():
    assert gpu_name_to_registry_key("AMD MI300") == ""
    assert gpu_name_to_registry_key("") == ""
    assert gpu_name_to_registry_key("GeForce RTX 4090") == ""
