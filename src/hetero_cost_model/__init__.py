"""Heterogeneous LLM inference cost model.

Subpackages
-----------
- ``hardware``    : GPU hardware registry and feature vectors.
- ``graph``       : torch.fx-based computation-graph extraction.
- ``strategies``  : inference-config helpers (batch size / sequence length).
- ``profiling``   : wall-clock latency measurement primitives.
- ``data``        : (graph, config, hardware, latency) → PyG dataset.
- ``models``      : GNN / Graph-Transformer / MLP cost-model architectures.
- ``training``    : training loop, losses, and hyper-parameter config.
- ``baselines``   : Roofline analytical model.
- ``metrics``     : MAPE / Spearman ρ / top-k / NDCG evaluation metrics.

The top-level package intentionally does *not* eagerly import submodules that
depend on ``torch_geometric``, so lightweight utilities (``hardware``, ``graph``,
``metrics``) remain importable even in environments without PyG.
"""

__version__ = "0.1.0"
