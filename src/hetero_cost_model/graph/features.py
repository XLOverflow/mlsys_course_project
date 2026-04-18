"""Typed node and graph representations, decoupled from any tracer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

from hetero_cost_model.graph.shapes import Shape, numel
from hetero_cost_model.graph.vocab import (
    NODE_FEATURE_DIM,
    NUM_OP_TYPES,
    OP_TYPE_TO_IDX,
)


@dataclass
class NodeFeature:
    """A single graph node annotated with the features the cost model needs."""

    name: str
    op_type: str
    input_shapes: List[Shape]
    output_shape: Shape
    dtype: str
    flops: float
    memory_bytes: float

    def to_vector(self) -> List[float]:
        vec = [0.0] * NUM_OP_TYPES
        idx = OP_TYPE_TO_IDX.get(self.op_type, OP_TYPE_TO_IDX["unknown"])
        vec[idx] = 1.0
        vec.append(math.log1p(max(self.flops, 0.0)))
        vec.append(math.log1p(max(self.memory_bytes, 0.0)))
        vec.append(math.log1p(numel(self.output_shape)))
        assert len(vec) == NODE_FEATURE_DIM
        return vec


@dataclass
class GraphRepr:
    """A framework-agnostic DAG of :class:`NodeFeature` plus edge list."""

    nodes: List[NodeFeature]
    edges: List[Tuple[int, int]]
    name: str = ""

    def num_nodes(self) -> int:
        return len(self.nodes)

    def node_feature_matrix(self) -> List[List[float]]:
        return [n.to_vector() for n in self.nodes]

    def total_flops(self) -> float:
        return sum(n.flops for n in self.nodes)

    def total_memory(self) -> float:
        return sum(n.memory_bytes for n in self.nodes)


# --- Graph-level "global summary" features ----------------------------------
#
# These four values mirror what XGBoost gets as tabular input (log1p of the
# total compute / memory / node count / edge count). Used as a residual "skip
# connection" into the GNN head so that the model has direct access to the
# Roofline-style summary statistics; message passing then only needs to learn
# the *residual* on top, not reconstruct scale from per-node features.

GRAPH_GLOBAL_FEATURE_DIM: int = 4


def graph_global_features(g: GraphRepr) -> List[float]:
    """Four log1p scalars: total_flops, total_memory_bytes, num_nodes, num_edges."""
    return [
        math.log1p(max(g.total_flops(), 0.0)),
        math.log1p(max(g.total_memory(), 0.0)),
        math.log1p(g.num_nodes()),
        math.log1p(len(g.edges)),
    ]


__all__ = [
    "NodeFeature", "GraphRepr",
    "GRAPH_GLOBAL_FEATURE_DIM", "graph_global_features",
]
