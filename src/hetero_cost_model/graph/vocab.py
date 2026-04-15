"""Operator type vocabulary for computation-graph node features.

Node feature layout
-------------------
    [ op-type one-hot (NUM_OP_TYPES) | log_flops | log_memory | log_numel ]

Total width = :data:`NODE_FEATURE_DIM`.
"""
from typing import Dict, Tuple


OP_TYPES: Tuple[str, ...] = (
    "placeholder", "output",
    "linear", "matmul", "conv", "attention", "embedding",
    "layernorm", "batchnorm", "softmax",
    "gelu", "relu", "silu", "tanh",
    "add", "mul", "sub", "div", "mean", "sum",
    "reshape", "transpose", "cat", "split", "slice",
    "dropout", "unknown",
)

OP_TYPE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(OP_TYPES)}

NUM_OP_TYPES: int = len(OP_TYPES)

# Three extra continuous scalars appended to each node vector.
_NUM_SCALAR_FEATURES = 3

NODE_FEATURE_DIM: int = NUM_OP_TYPES + _NUM_SCALAR_FEATURES


__all__ = ["OP_TYPES", "OP_TYPE_TO_IDX", "NUM_OP_TYPES", "NODE_FEATURE_DIM"]
