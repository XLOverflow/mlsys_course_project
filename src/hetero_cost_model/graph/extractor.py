"""Trace a PyTorch module with torch.fx and produce a :class:`GraphRepr`."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from hetero_cost_model.graph.features import GraphRepr, NodeFeature
from hetero_cost_model.graph.flops import estimate_flops
from hetero_cost_model.graph.shapes import Shape, numel


_MODULE_CLASS_KEYWORDS = (
    "linear", "layernorm", "batchnorm", "embedding", "conv",
    "attention", "dropout", "gelu", "relu", "silu", "tanh", "softmax",
)

_FUNCTION_KEYWORDS = (
    "matmul", "layer_norm", "softmax", "gelu", "relu", "silu", "tanh",
    "embedding", "cat", "split", "reshape", "view", "transpose", "permute",
    "mean", "sum", "add", "mul", "sub", "div",
)

_FUNCTION_ALIASES = {
    "layer_norm": "layernorm",
    "view": "reshape",
    "permute": "transpose",
}


def extract_graph(
    model: torch.nn.Module,
    example_inputs,
    name: str = "",
) -> GraphRepr:
    """Symbolically trace ``model`` and return a feature-annotated graph."""
    traced: fx.GraphModule = fx.symbolic_trace(model)
    args = example_inputs if isinstance(example_inputs, (list, tuple)) else (example_inputs,)
    try:
        ShapeProp(traced).propagate(*args)
    except Exception:
        # Shape prop is best-effort — some ops in HF models reject it.
        pass

    fx_nodes = list(traced.graph.nodes)
    name_to_idx = {n.name: i for i, n in enumerate(fx_nodes)}

    nodes = [_build_node_feature(n, traced) for n in fx_nodes]
    edges: List[Tuple[int, int]] = [
        (name_to_idx[src.name], dst_i)
        for dst_i, n in enumerate(fx_nodes)
        for src in n.all_input_nodes
    ]
    return GraphRepr(nodes=nodes, edges=edges, name=name)


# --- Node construction ---------------------------------------------------------

def _build_node_feature(fx_node: fx.Node, module: torch.nn.Module) -> NodeFeature:
    op_type = _classify(fx_node, module)
    out_shape, dtype = _read_meta(fx_node)
    in_shapes = [s for s in (_read_meta(a)[0] for a in fx_node.all_input_nodes) if s]
    flops = estimate_flops(op_type, in_shapes, out_shape)
    memory_bytes = numel(out_shape) * 4  # assume fp32; refined at profiling time
    return NodeFeature(
        name=fx_node.name,
        op_type=op_type,
        input_shapes=in_shapes,
        output_shape=out_shape,
        dtype=dtype,
        flops=flops,
        memory_bytes=memory_bytes,
    )


def _read_meta(node: fx.Node) -> Tuple[Shape, str]:
    meta = getattr(node, "meta", {}).get("tensor_meta") if hasattr(node, "meta") else None
    if meta is None:
        return (), "float32"
    try:
        return tuple(int(d) for d in meta.shape), str(meta.dtype)
    except Exception:
        return (), "float32"


def _classify(node: fx.Node, module: torch.nn.Module) -> str:
    if node.op == "placeholder":
        return "placeholder"
    if node.op == "output":
        return "output"
    if node.op == "call_module":
        return _classify_module(node, module)
    if node.op in ("call_function", "call_method"):
        return _classify_function(node)
    return "unknown"


def _classify_module(node: fx.Node, module: torch.nn.Module) -> str:
    try:
        submod = module.get_submodule(node.target)
    except AttributeError:
        return "unknown"
    cls = type(submod).__name__.lower()
    for key in _MODULE_CLASS_KEYWORDS:
        if key in cls:
            return key
    return "unknown"


def _classify_function(node: fx.Node) -> str:
    name = str(node.target).lower()
    for key in _FUNCTION_KEYWORDS:
        if key in name:
            return _FUNCTION_ALIASES.get(key, key)
    return "unknown"


__all__ = ["extract_graph"]
