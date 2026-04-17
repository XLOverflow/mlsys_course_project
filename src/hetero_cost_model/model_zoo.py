"""Registry of HF models targeted by this project + loading helpers.

Keeps the model list, load semantics, and fx input-name contract in ONE
place so that smoke tests, graph pre-extraction, and the profiling driver
all agree on what's being measured.

All models are loaded with ``attn_implementation="eager"`` for two reasons:

  1. Flash-Attention-2 is not supported on V100, and we need a single
     attention kernel path across V100→B200 for fair cross-GPU comparison.
  2. The sdpa mask path in ``transformers >= 4.52`` uses ``torch.vmap``,
     which HF's fx tracer (HFProxy) can't proxy. Eager avoids that call.

See also: requirements.txt pin ``transformers>=4.35,<4.52``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch


@dataclass(frozen=True)
class ModelSpec:
    """Static description of one target model."""

    name: str      # CSV-friendly short name used across the project
    hf_id: str     # HuggingFace hub identifier for ``from_pretrained``
    family: str    # "decoder" | "encoder" | "enc-dec"

    # Architectural maximum input length (== ``max_position_embeddings``
    # for models with learned absolute position embeddings). Configs with
    # ``seq_len > max_seq_len`` should be pre-filtered from the profiling
    # grid — running them produces a model-capability failure, NOT a
    # hardware OOM, and carries no signal about hardware feasibility.
    # GPT-2 and T5 are effectively unbounded for our 1024-length grid.
    max_seq_len: int = 10**9


MODELS: List[ModelSpec] = [
    ModelSpec("gpt2-small",  "gpt2",               "decoder",  max_seq_len=1024),
    ModelSpec("gpt2-medium", "gpt2-medium",        "decoder",  max_seq_len=1024),
    ModelSpec("gpt2-large",  "gpt2-large",         "decoder",  max_seq_len=1024),
    ModelSpec("bert-base",   "bert-base-uncased",  "encoder",  max_seq_len=512),
    ModelSpec("bert-large",  "bert-large-uncased", "encoder",  max_seq_len=512),
    ModelSpec("t5-small",    "t5-small",           "enc-dec",  max_seq_len=1024),
]

MODEL_BY_NAME: Dict[str, ModelSpec] = {m.name: m for m in MODELS}


def load_model(spec: ModelSpec) -> torch.nn.Module:
    """Load a model with eager attention (mandatory project default)."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        T5ForConditionalGeneration,
    )
    kwargs = {"attn_implementation": "eager"}
    if spec.family == "decoder":
        return AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    if spec.family == "encoder":
        return AutoModelForMaskedLM.from_pretrained(spec.hf_id, **kwargs)
    if spec.family == "enc-dec":
        return T5ForConditionalGeneration.from_pretrained(spec.hf_id, **kwargs)
    raise ValueError(f"unknown family: {spec.family}")


def example_inputs(
    spec: ModelSpec, batch_size: int = 1, seq_len: int = 32,
) -> Tuple[torch.Tensor, ...]:
    """Random integer token tensors matching each family's forward signature."""
    ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    if spec.family == "enc-dec":
        decoder_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
        return (ids, decoder_ids)
    return (ids,)


def hf_input_names(spec: ModelSpec) -> List[str]:
    """Input names used by ``transformers.utils.fx.symbolic_trace``."""
    if spec.family == "enc-dec":
        return ["input_ids", "decoder_input_ids"]
    return ["input_ids"]


__all__ = [
    "ModelSpec", "MODELS", "MODEL_BY_NAME",
    "load_model", "example_inputs", "hf_input_names",
]
