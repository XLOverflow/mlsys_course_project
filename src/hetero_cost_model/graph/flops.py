"""Heuristic FLOPs estimation per op type.

These estimates are deliberately rough — they become input features, not
the prediction target. The learned cost model uses them as a hint alongside
the graph topology and is responsible for the final latency prediction.
"""
from typing import List, Sequence

from hetero_cost_model.graph.shapes import Shape, numel


_ELEMENTWISE = frozenset({
    "layernorm", "batchnorm", "softmax",
    "gelu", "relu", "silu", "tanh", "dropout",
    "add", "mul", "sub", "div", "mean", "sum",
})


def estimate_flops(
    op_type: str,
    input_shapes: Sequence[Shape],
    output_shape: Shape,
) -> float:
    """Return a rough FLOPs estimate for a single op."""
    out_n = numel(output_shape)
    if op_type in ("linear", "matmul"):
        if input_shapes and input_shapes[0]:
            inner_dim = input_shapes[0][-1]
            return 2.0 * out_n * inner_dim
        return 2.0 * out_n
    if op_type == "conv":
        return 2.0 * out_n * 9  # 3x3 kernel approximation
    if op_type == "attention":
        return 4.0 * out_n
    if op_type in _ELEMENTWISE:
        return float(out_n)
    return 0.0


__all__ = ["estimate_flops"]
