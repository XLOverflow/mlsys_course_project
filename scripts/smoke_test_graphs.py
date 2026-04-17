"""Day 1 smoke test: extract computation graphs for all 6 target models.

Hard blocker before any GPU profiling. If any model fails to trace here,
we need to decide (a) drop that model, (b) switch to ONNX export, or
(c) refactor extract_graph. Running on CPU only — no GPU needed.

Usage:
    python scripts/smoke_test_graphs.py
"""
from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch

from hetero_cost_model.graph import extract_graph


BATCH_SIZE = 1
SEQ_LEN = 32   # small to keep trace fast; shapes don't affect trace correctness


@dataclass
class ModelSpec:
    name: str             # CSV-friendly short name
    hf_id: str            # HF hub identifier
    family: str           # "decoder" | "encoder" | "enc-dec"


MODELS: List[ModelSpec] = [
    ModelSpec("gpt2-small",  "gpt2",               "decoder"),
    ModelSpec("gpt2-medium", "gpt2-medium",        "decoder"),
    ModelSpec("gpt2-large",  "gpt2-large",         "decoder"),
    ModelSpec("bert-base",   "bert-base-uncased",  "encoder"),
    ModelSpec("bert-large",  "bert-large-uncased", "encoder"),
    ModelSpec("t5-small",    "t5-small",           "enc-dec"),
]


def _load_model(spec: ModelSpec) -> torch.nn.Module:
    """Load each model with the minimum class that preserves the full graph.

    ``attn_implementation="eager"`` is mandatory: (a) project decision (see
    two_week_execution_plan.md §1, Flash-Attention-2 is unsupported on V100),
    (b) the new ``sdpa`` path in transformers ≥ 4.55 uses ``torch.vmap`` which
    HF's fx tracer cannot proxy.
    """
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        T5ForConditionalGeneration,
    )
    kw = {"attn_implementation": "eager"}
    if spec.family == "decoder":
        return AutoModelForCausalLM.from_pretrained(spec.hf_id, **kw)
    if spec.family == "encoder":
        return AutoModelForMaskedLM.from_pretrained(spec.hf_id, **kw)
    if spec.family == "enc-dec":
        return T5ForConditionalGeneration.from_pretrained(spec.hf_id, **kw)
    raise ValueError(f"unknown family: {spec.family}")


def _example_inputs(spec: ModelSpec) -> Tuple[torch.Tensor, ...]:
    input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
    if spec.family == "enc-dec":
        decoder_input_ids = torch.randint(0, 1000, (BATCH_SIZE, SEQ_LEN), dtype=torch.long)
        return (input_ids, decoder_input_ids)
    return (input_ids,)


def _input_names(spec: ModelSpec) -> List[str]:
    if spec.family == "enc-dec":
        return ["input_ids", "decoder_input_ids"]
    return ["input_ids"]


def _try_extract(spec: ModelSpec) -> Tuple[bool, str]:
    try:
        model = _load_model(spec)
        model.eval()
        inputs = _example_inputs(spec)
        graph = extract_graph(
            model, inputs, name=spec.name, hf_input_names=_input_names(spec),
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
