"""Profile all (model × batch × seq) configurations on the current GPU.

Writes one row per config to a CSV matching the schema in
``data_collection_plan.md §3.1``. Designed to be **idempotent** (resumes
after Modal preemption) and **OOM-safe** (logs an OOM row and continues).

Typical invocation:

  python scripts/run_profiling.py \\
      --gpu v100 \\
      --models gpt2-small gpt2-medium bert-base bert-large t5-small \\
      --batch-sizes 1 2 4 8 \\
      --seq-lens 64 128 256 \\
      --output data/raw/v100.csv

Models use ``attn_implementation="eager"`` (see model_zoo.py).
"""
from __future__ import annotations

import argparse
import csv
import platform as _platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import torch

from hetero_cost_model.model_zoo import MODEL_BY_NAME, ModelSpec, load_model
from hetero_cost_model.profiling import profile_callable
from hetero_cost_model.runtime_info import current_gpu_info


CSV_COLUMNS: List[str] = [
    # Identifiers / config
    "model_name", "gpu", "batch_size", "seq_len",
    # Latency stats (p50_ms is the training target)
    "mean_ms", "std_ms", "p50_ms", "p95_ms", "n_runs", "noisy",
    # Memory / OOM
    "peak_memory_mb",
    # Actual hardware (guard against Modal SKU upgrade)
    "actual_gpu_name", "actual_mem_gb", "actual_sm_count",
    # Settings
    "attn_impl",
    # Environment provenance
    "platform", "cuda_version", "driver_version", "torch_version", "timestamp",
]


# --- Driver ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gpu", required=True,
                   help="Declared GPU label (v100/a100/h100/h200/b200). "
                        "Also written to CSV; actual_gpu_name is queried at runtime.")
    p.add_argument("--models", nargs="+", required=True,
                   help="Model names from model_zoo (e.g. gpt2-small bert-base)")
    p.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    p.add_argument("--seq-lens", nargs="+", type=int, required=True)
    p.add_argument("--output", type=Path, required=True, help="CSV output path")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--runs", type=int, default=100)
    p.add_argument("--device", default="cuda:0", help="PyTorch device string")
    p.add_argument("--force", action="store_true",
                   help="Re-profile configs that already have rows in --output")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no CUDA device visible; timing will use perf_counter. "
              "Useful for dry-run only.")

    specs = [_resolve_spec(name) for name in args.models]
    gpu_info = current_gpu_info()

    print(f"Target device: {device}")
    print(f"  declared gpu    : {args.gpu}")
    print(f"  actual_gpu_name : {gpu_info.actual_gpu_name}")
    print(f"  registry_key    : {gpu_info.registry_key or '(not in registry)'}")
    print(f"  actual_mem_gb   : {gpu_info.actual_mem_gb:.1f}")
    print(f"  actual_sm_count : {gpu_info.actual_sm_count}")
    if gpu_info.registry_key and gpu_info.registry_key != args.gpu:
        print(
            f"  ⚠️  SKU mismatch: declared '{args.gpu}' but runtime sees "
            f"'{gpu_info.registry_key}' (Modal auto-upgrade?). "
            f"Training will dispatch by actual_gpu_name; declared label is for bookkeeping only."
        )

    configs = _config_grid(specs, args.batch_sizes, args.seq_lens)
    print(f"\nConfig grid: {len(configs)} total points "
          f"({len(specs)} models × {len(args.batch_sizes)} batches × {len(args.seq_lens)} seqs)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.force else _existing_rows(args.output)
    if done:
        print(f"Resume: {len(done)} rows already exist in {args.output} — skipping")
    writer = _CsvWriter(args.output)

    try:
        for spec, bs, sl in configs:
            key = (spec.name, bs, sl, gpu_info.actual_gpu_name)
            if key in done:
                continue
            row = _profile_one(
                spec, bs, sl, device,
                gpu_label=args.gpu, gpu_info=gpu_info,
                warmup=args.warmup, runs=args.runs,
            )
            writer.write(row)
    finally:
        writer.close()

    print(f"\nDone. CSV at {args.output}")
    return 0


# --- Per-config profiling ----------------------------------------------------

def _profile_one(
    spec: ModelSpec, batch_size: int, seq_len: int, device: str,
    *, gpu_label: str, gpu_info, warmup: int, runs: int,
) -> Dict[str, object]:
    tag = f"{spec.name} bs={batch_size} sl={seq_len}"
    print(f"[...] {tag}")
    base = _base_row(spec, batch_size, seq_len, gpu_label, gpu_info)

    try:
        model = load_model(spec).to(device).half().eval()
    except Exception as e:
        print(f"[SKIP] {tag} load failed: {type(e).__name__}: {e}")
        return {**base, "n_runs": 0, "noisy": False, "peak_memory_mb": float("nan")}

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long, device=device)
    extra = {}
    if spec.family == "enc-dec":
        extra["decoder_input_ids"] = torch.randint(
            0, 1000, (batch_size, seq_len), dtype=torch.long, device=device,
        )

    try:
        torch.cuda.reset_peak_memory_stats() if device.startswith("cuda") else None

        def step() -> None:
            with torch.inference_mode():
                model(input_ids, **extra)

        result = profile_callable(step, warmup=warmup, runs=runs, device=device)
        peak_mem_mb = (
            torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else 0.0
        )
        print(
            f"[OK]  {tag} "
            f"p50={result.p50_ms:7.3f}ms mean={result.mean_ms:7.3f}ms "
            f"std={result.std_ms:.3f}ms peak_mem={peak_mem_mb:.0f}MB "
            f"noisy={result.noisy}"
        )
        return {
            **base,
            **result.as_dict(),
            "peak_memory_mb": peak_mem_mb,
        }
    except torch.cuda.OutOfMemoryError:
        print(f"[OOM] {tag}")
        torch.cuda.empty_cache() if device.startswith("cuda") else None
        return {
            **base,
            "mean_ms": float("nan"), "std_ms": float("nan"),
            "p50_ms": float("nan"), "p95_ms": float("nan"),
            "n_runs": 0, "noisy": False,
            "peak_memory_mb": float("nan"),
        }
    except Exception as e:
        print(f"[ERR] {tag} {type(e).__name__}: {e}")
        return {
            **base,
            "mean_ms": float("nan"), "std_ms": float("nan"),
            "p50_ms": float("nan"), "p95_ms": float("nan"),
            "n_runs": 0, "noisy": False,
            "peak_memory_mb": float("nan"),
        }
    finally:
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()


def _base_row(
    spec: ModelSpec, batch_size: int, seq_len: int, gpu_label: str, gpu_info,
) -> Dict[str, object]:
    return {
        "model_name": spec.name,
        "gpu": gpu_label,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "actual_gpu_name": gpu_info.actual_gpu_name,
        "actual_mem_gb": round(gpu_info.actual_mem_gb, 2),
        "actual_sm_count": gpu_info.actual_sm_count,
        "attn_impl": "eager",
        "platform": _platform.node(),
        "cuda_version": gpu_info.cuda_version,
        "driver_version": gpu_info.driver_version,
        "torch_version": gpu_info.torch_version,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --- CSV helpers (idempotent append) -----------------------------------------

def _existing_rows(path: Path) -> Set[Tuple[str, int, int, str]]:
    """Rows already in the CSV, keyed by (model, bs, sl, actual_gpu_name)."""
    if not path.exists():
        return set()
    done: Set[Tuple[str, int, int, str]] = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add((
                    row["model_name"],
                    int(row["batch_size"]),
                    int(row["seq_len"]),
                    row["actual_gpu_name"],
                ))
            except (KeyError, ValueError):
                continue
    return done


class _CsvWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._new_file = not path.exists() or path.stat().st_size == 0
        self._fh = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS)
        if self._new_file:
            self._writer.writeheader()

    def write(self, row: Dict[str, object]) -> None:
        self._writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})
        self._fh.flush()   # flush per row so preemption doesn't lose buffered data

    def close(self) -> None:
        self._fh.close()


# --- Misc --------------------------------------------------------------------

def _resolve_spec(name: str) -> ModelSpec:
    if name not in MODEL_BY_NAME:
        raise SystemExit(
            f"unknown model: {name}. Known: {', '.join(MODEL_BY_NAME)}"
        )
    return MODEL_BY_NAME[name]


def _config_grid(
    specs: Iterable[ModelSpec], batch_sizes: List[int], seq_lens: List[int],
) -> List[Tuple[ModelSpec, int, int]]:
    return [(spec, bs, sl) for spec in specs for bs in batch_sizes for sl in seq_lens]


if __name__ == "__main__":
    sys.exit(main())
