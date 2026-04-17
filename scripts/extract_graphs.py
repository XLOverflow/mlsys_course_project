"""Pre-extract computation graphs for all target models and serialize to pkl.

Training reads these once from ``data/graphs/{model_name}.pkl`` instead of
re-tracing every epoch — tracing gpt2-large takes ~5s, ``pickle.load``
takes <50 ms.

Run after ``scripts/smoke_test_graphs.py`` is green. CPU only, no GPU needed.

Usage:
    python scripts/extract_graphs.py                 # only extract missing pkls
    python scripts/extract_graphs.py --force         # re-extract all
    python scripts/extract_graphs.py --out-dir DIR   # custom output dir
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

from hetero_cost_model.graph import extract_graph
from hetero_cost_model.model_zoo import (
    MODELS,
    ModelSpec,
    example_inputs,
    hf_input_names,
    load_model,
)


DEFAULT_OUT_DIR = Path("data/graphs")
BATCH_SIZE = 1
SEQ_LEN = 32   # trace result is batch/seq-agnostic; use small for speed


def _extract_one(spec: ModelSpec, out_path: Path) -> None:
    model = load_model(spec)
    model.eval()
    inputs = example_inputs(spec, BATCH_SIZE, SEQ_LEN)
    graph = extract_graph(
        model, inputs, name=spec.name, hf_input_names=hf_input_names(spec),
    )
    with open(out_path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[OK]   {spec.name:14s} "
        f"nodes={graph.num_nodes():4d}  "
        f"edges={len(graph.edges):5d}  "
        f"flops={graph.total_flops():.2e}  "
        f"mem={graph.total_memory() / 1e6:6.1f}MB  "
        f"→ {out_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force", action="store_true", help="re-extract even if pkl exists")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {len(MODELS)} graphs to {args.out_dir}/")
    print(f"  (set --force to overwrite)\n")

    failed: List[str] = []
    for spec in MODELS:
        out_path = args.out_dir / f"{spec.name}.pkl"
        if out_path.exists() and not args.force:
            size_kb = out_path.stat().st_size / 1024
            print(f"[SKIP] {spec.name:14s} exists ({size_kb:.1f}KB at {out_path})")
            continue
        print(f"[...]  {spec.name:14s} loading + tracing...")
        try:
            _extract_one(spec, out_path)
        except Exception as e:
            print(f"[FAIL] {spec.name:14s} {type(e).__name__}: {e}")
            failed.append(spec.name)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        print("Run `python scripts/smoke_test_graphs.py` for full traceback.")
        return 1
    print(f"All {len(MODELS)} graphs ready at {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
