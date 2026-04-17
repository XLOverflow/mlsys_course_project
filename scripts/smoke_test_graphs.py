"""Day 1 smoke test: extract computation graphs for all target models.

Hard blocker before any GPU profiling. If any model fails to trace here,
decide (a) drop that model, (b) switch to ONNX export, or (c) refactor
``extract_graph``. Running on CPU only — no GPU needed.

Usage:
    python scripts/smoke_test_graphs.py
"""
from __future__ import annotations

import sys
import traceback
from typing import List, Tuple

from hetero_cost_model.graph import extract_graph
from hetero_cost_model.model_zoo import (
    MODELS,
    ModelSpec,
    example_inputs,
    hf_input_names,
    load_model,
)


BATCH_SIZE = 1
SEQ_LEN = 32   # small to keep trace fast; shapes don't affect trace correctness


def _try_extract(spec: ModelSpec) -> Tuple[bool, str]:
    try:
        model = load_model(spec)
        model.eval()
        inputs = example_inputs(spec, BATCH_SIZE, SEQ_LEN)
        graph = extract_graph(
            model, inputs, name=spec.name, hf_input_names=hf_input_names(spec),
        )
        msg = (
            f"nodes={graph.num_nodes():4d}  "
            f"edges={len(graph.edges):5d}  "
            f"total_flops={graph.total_flops():.2e}  "
            f"total_mem={graph.total_memory() / 1e6:.1f} MB"
        )
        return True, msg
    except Exception as e:
        return False, f"{type(e).__name__}: {e}\n{traceback.format_exc()}"


def main() -> int:
    print(f"Day 1 fx-tracing smoke test on {len(MODELS)} models")
    print(f"  batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, device=cpu\n")

    results: List[Tuple[ModelSpec, bool, str]] = []
    for spec in MODELS:
        print(f"[...] {spec.name:14s} ({spec.hf_id})")
        ok, msg = _try_extract(spec)
        results.append((spec, ok, msg))
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {spec.name:14s} {msg if ok else msg.splitlines()[0]}")

    print("\n" + "=" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = [s for s, ok, _ in results if not ok]
    print(f"Summary: {passed}/{len(MODELS)} passed")
    if failed:
        print(f"FAILED: {', '.join(s.name for s in failed)}")
        print("\nFull tracebacks for failed models:")
        for spec, ok, msg in results:
            if not ok:
                print(f"\n--- {spec.name} ---")
                print(msg)
        return 1
    print("ALL GREEN — unblocked for Day 3 profiling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
