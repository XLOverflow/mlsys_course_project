"""PyG dataset construction from (graph, config, hardware, latency) samples.

Each sample becomes one PyG ``Data`` object:
  - ``x``  : node feature matrix [N, NODE_FEATURE_DIM] — op structure only
  - ``s``  : inference config vector [CONFIG_FEATURE_DIM] — graph-level
  - ``h``  : hardware feature vector [HARDWARE_FEATURE_DIM] — graph-level
  - ``y``  : measured latency in ms — scalar

s and h are graph-level attributes concatenated in the model head after
graph readout, so node features stay architecture-only.

``load_samples_from_csv`` ties together the three persisted artifacts into
``Sample`` objects ready for training:

  * profiling CSV        (produced by ``scripts/run_profiling.py``)
  * pre-extracted graphs (produced by ``scripts/extract_graphs.py``)
  * ``HARDWARE_REGISTRY`` (in-memory registry keyed by ``actual_gpu_name``)
"""
from __future__ import annotations

import csv
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch_geometric.data import Data, Dataset

from hetero_cost_model.graph import (
    GRAPH_GLOBAL_FEATURE_DIM,
    GraphRepr,
    graph_global_features,
)
from hetero_cost_model.hardware import HARDWARE_REGISTRY, Hardware
from hetero_cost_model.runtime_info import gpu_name_to_registry_key
from hetero_cost_model.strategies import InferenceConfig


@dataclass
class Sample:
    """A single training example."""

    graph: GraphRepr
    config: InferenceConfig
    hardware: Hardware
    latency_ms: float
    model_name: str = ""


def sample_to_pyg(sample: Sample) -> Data:
    """Convert one :class:`Sample` into a PyG ``Data`` object.

    Attached graph-level attributes:

    - ``s`` : inference config [CONFIG_FEATURE_DIM] — (batch, seq) normalized
    - ``h`` : hardware features [HARDWARE_FEATURE_DIM] — 5-dim spec vector
    - ``g`` : graph-global summary [GRAPH_GLOBAL_FEATURE_DIM] — log1p of
      (total_flops, total_memory_bytes, num_nodes, num_edges). Mirrors the
      tabular features XGBoost gets; models can optionally concat ``g`` to
      their readout so they don't have to re-derive totals from per-node
      features.
    """
    x = torch.tensor(sample.graph.node_feature_matrix(), dtype=torch.float)
    s = torch.tensor(sample.config.to_vector(), dtype=torch.float)
    h = torch.tensor(sample.hardware.to_vector(), dtype=torch.float)
    g = torch.tensor(graph_global_features(sample.graph), dtype=torch.float)
    y = torch.tensor([sample.latency_ms], dtype=torch.float)

    if sample.graph.edges:
        edge_index = torch.tensor(sample.graph.edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)

    return Data(x=x, edge_index=edge_index, s=s, h=h, g=g, y=y)


class LatencyDataset(Dataset):
    """In-memory PyG dataset — sufficient for the project's ~1500 samples."""

    def __init__(self, samples: List[Sample]):
        super().__init__()
        self._cache = [sample_to_pyg(s) for s in samples]

    def len(self) -> int:
        return len(self._cache)

    def get(self, idx: int) -> Data:
        return self._cache[idx]

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> Data:
        return self._cache[idx]


def load_samples_from_csv(
    csv_path: Path,
    graph_dir: Path,
    *,
    graphs: Optional[Dict[str, GraphRepr]] = None,
) -> List[Sample]:
    """Parse a profiling CSV into training samples.

    Skips rows that are OOM (``p50_ms`` NaN or ``n_runs == 0``) or that
    reference an unknown ``actual_gpu_name`` / ``model_name``.

    Hardware is looked up by ``actual_gpu_name`` (runtime-queried), not the
    declared ``gpu`` label — this isolates training from Modal SKU upgrades.

    Parameters
    ----------
    csv_path
        CSV produced by ``scripts/run_profiling.py``.
    graph_dir
        Directory containing ``<model_name>.pkl`` from ``extract_graphs.py``.
    graphs
        Optional pre-loaded graph cache; built on demand if ``None``.
    """
    graph_dir = Path(graph_dir)
    cache: Dict[str, GraphRepr] = dict(graphs) if graphs is not None else {}
    samples: List[Sample] = []

    with open(csv_path) as fh:
        for row in csv.DictReader(fh):
            if not _row_is_measurement(row):
                continue

            model_name = row["model_name"]
            graph = cache.get(model_name)
            if graph is None:
                pkl = graph_dir / f"{model_name}.pkl"
                if not pkl.exists():
                    continue
                with open(pkl, "rb") as gh:
                    graph = pickle.load(gh)
                cache[model_name] = graph

            key = _resolve_registry_key(row)
            if key is None:
                continue

            samples.append(Sample(
                graph=graph,
                config=InferenceConfig(int(row["batch_size"]), int(row["seq_len"])),
                hardware=HARDWARE_REGISTRY[key],
                latency_ms=float(row["p50_ms"]),
                model_name=model_name,
            ))
    return samples


def _row_is_measurement(row: Dict[str, str]) -> bool:
    try:
        p50 = float(row.get("p50_ms", "nan"))
    except ValueError:
        return False
    if math.isnan(p50):
        return False
    try:
        n_runs = int(row.get("n_runs", "0"))
    except ValueError:
        return False
    return n_runs > 0


def _resolve_registry_key(row: Dict[str, str]) -> Optional[str]:
    """Prefer the runtime-observed device name; fall back to declared label."""
    actual = row.get("actual_gpu_name", "") or ""
    key = gpu_name_to_registry_key(actual)
    if key and key in HARDWARE_REGISTRY:
        return key
    # Fallback: if the CSV was written with the lowercase registry key directly
    # (e.g. by the smoke test), use it.
    if actual in HARDWARE_REGISTRY:
        return actual
    declared = row.get("gpu", "")
    if declared in HARDWARE_REGISTRY:
        return declared
    return None


__all__ = ["Sample", "LatencyDataset", "sample_to_pyg", "load_samples_from_csv"]
