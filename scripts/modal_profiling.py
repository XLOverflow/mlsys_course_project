"""Dispatch profiling to Modal on a locked GPU SKU.

Wraps ``scripts/run_profiling.py`` in a Modal function so we can profile on
A100-40GB / H200 / B200 without provisioning a real GPU locally. The SKU
is pinned via Modal's ``!`` / exact-name syntax to prevent silent upgrades
(e.g. ``A100`` landing on an 80 GB card).

Usage (local dispatch):

  modal run scripts/modal_profiling.py \\
      --gpu-sku a100-40gb \\
      --models bert-base \\
      --batch-sizes 1 \\
      --seq-lens 64 \\
      --output data/raw/a100_modal.csv

  # Full grid, all 6 models:
  modal run scripts/modal_profiling.py --gpu-sku a100-40gb --models all \\
      --batch-sizes 1,2,4,8,16 --seq-lens 64,128,256,512
"""
from __future__ import annotations

from pathlib import Path

import modal


# --- Container image ---------------------------------------------------------

# Single unified image. torch>=2.7 is the minimum version that ships kernels
# for Blackwell (sm_100 / B200). We use the same PyTorch binary on every GPU
# (Turing → Hopper → Blackwell) so cross-GPU latency comparisons don't have
# a software-version confound — the "B200 zero-shot" numbers land against
# anchors profiled with the exact same kernels.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7",
        "torch-geometric>=2.4",
        "transformers>=4.35,<4.52",               # see requirements.txt rationale
        "numpy", "pandas", "scipy", "scikit-learn", "tqdm",
    )
    # Mount source + scripts at runtime; no rebuild when code changes.
    .add_local_dir("src",     remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml")
)


app = modal.App("hetero-cost-profiling", image=IMAGE)

# Persistent Volume so CSV writes survive container preemption. Modal T4
# workloads are preemptible and can be reclaimed mid-session; without
# persistence the run_profiling.py driver has to start from scratch every
# time (losing all flushed rows). With the volume, the driver's idempotent
# resume logic (skip rows already in the CSV) means each restart picks up
# where the last one died.
CSV_VOLUME = modal.Volume.from_name(
    "hetero-cost-profiling-csvs", create_if_missing=True,
)


# --- Per-SKU Modal function entries ------------------------------------------
#
# We register one @app.function per locked SKU so the Modal scheduler can't
# auto-upgrade us to a bigger card. Each delegates to ``_run_profiling``.

def _run_profiling(
    gpu_label: str,
    models: list[str],
    batch_sizes: list[int],
    seq_lens: list[int],
    warmup: int,
    runs: int,
    out_basename: str | None = None,
    mode: str = "prefill",
) -> bytes:
    """Install the local package, run the profiling driver, return CSV bytes.

    CSV is written into ``/csvs`` (Modal Volume) so progress survives
    preemption. The driver uses its own idempotent resume (``_existing_rows``)
    to skip configs already present in the CSV, turning a preempt → retry
    cycle into incremental checkpointing.

    Pass ``mode="decode"`` to time per-token autoregressive decode
    latency instead of full-prompt prefill latency. Encoder-only models
    are auto-skipped in decode mode by run_profiling.py.
    """
    import os
    import subprocess
    import sys

    os.chdir("/root")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/root", "--quiet"]
    )

    filename = out_basename or f"{gpu_label}.csv"
    output_path = f"/csvs/{filename}"
    os.makedirs("/csvs", exist_ok=True)
    cmd = [
        sys.executable, "scripts/run_profiling.py",
        "--gpu", gpu_label,
        "--models", *models,
        "--batch-sizes", *[str(x) for x in batch_sizes],
        "--seq-lens", *[str(x) for x in seq_lens],
        "--warmup", str(warmup),
        "--runs", str(runs),
        "--mode", mode,
        "--output", output_path,
    ]
    subprocess.check_call(cmd)
    # Ensure the final write is committed to the Volume before returning.
    CSV_VOLUME.commit()
    with open(output_path, "rb") as f:
        return f.read()


_VOLUME_MOUNT = {"/csvs": CSV_VOLUME}
# modal.Retries: whenever Modal preempts the container, retry up to 10 times.
# Combined with the CSV-on-Volume resume logic in run_profiling.py, this
# turns preempt → retry into incremental checkpointing.
_PREEMPT_RETRIES = modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=0.0)


# T4 dropped from the project: Modal worker stability issues (silent stalls
# during prefill, container-startup hangs during decode) made it unreliable
# across data-collection passes.

@app.function(gpu="L4", timeout=3600, volumes=_VOLUME_MOUNT, retries=_PREEMPT_RETRIES)
def profile_l4(
    models: list[str], batch_sizes: list[int], seq_lens: list[int],
    warmup: int, runs: int, out_basename: str | None = None,
    mode: str = "prefill",
) -> bytes:
    return _run_profiling("l4", models, batch_sizes, seq_lens, warmup, runs, out_basename, mode)


@app.function(gpu="A10", timeout=3600, volumes=_VOLUME_MOUNT, retries=_PREEMPT_RETRIES)
def profile_a10(
    models: list[str], batch_sizes: list[int], seq_lens: list[int],
    warmup: int, runs: int, out_basename: str | None = None,
    mode: str = "prefill",
) -> bytes:
    return _run_profiling("a10", models, batch_sizes, seq_lens, warmup, runs, out_basename, mode)


@app.function(gpu="A100-40GB", timeout=3600, volumes=_VOLUME_MOUNT, retries=_PREEMPT_RETRIES)
def profile_a100_40gb(
    models: list[str], batch_sizes: list[int], seq_lens: list[int],
    warmup: int, runs: int, out_basename: str | None = None,
    mode: str = "prefill",
) -> bytes:
    return _run_profiling("a100", models, batch_sizes, seq_lens, warmup, runs, out_basename, mode)


@app.function(gpu="H100!", timeout=3600, volumes=_VOLUME_MOUNT, retries=_PREEMPT_RETRIES)
def profile_h100(
    models: list[str], batch_sizes: list[int], seq_lens: list[int],
    warmup: int, runs: int, out_basename: str | None = None,
    mode: str = "prefill",
) -> bytes:
    """``H100!`` forces Modal to actually give us H100 (no auto-upgrade to H200)."""
    return _run_profiling("h100", models, batch_sizes, seq_lens, warmup, runs, out_basename, mode)


# H200 dropped from the project: gpt2-large decode showed an anomalous
# slowdown (25 ms vs H100's 16 ms) despite more memory bandwidth — likely a
# Hopper+ kernel dispatch quirk we did not investigate. To keep the dataset
# physically clean, H200 is excluded from cost-model training and evaluation.

@app.function(gpu="B200", timeout=3600, volumes=_VOLUME_MOUNT, retries=_PREEMPT_RETRIES)
def profile_b200(
    models: list[str], batch_sizes: list[int], seq_lens: list[int],
    warmup: int, runs: int, out_basename: str | None = None,
    mode: str = "prefill",
) -> bytes:
    return _run_profiling("b200", models, batch_sizes, seq_lens, warmup, runs, out_basename, mode)


# --- Local entrypoint --------------------------------------------------------

_ALL_MODELS = [
    "gpt2-small", "gpt2-medium", "gpt2-large",
    "bert-base", "bert-large", "t5-small",
]


@app.local_entrypoint()
def main(
    gpu_sku: str = "a100-40gb",
    models: str = "bert-base",
    batch_sizes: str = "1",
    seq_lens: str = "64",
    warmup: int = 50,
    runs: int = 100,
    mode: str = "prefill",
    output: str = "",
):
    """Dispatch a profiling run and write the CSV locally.

    Set ``mode=decode`` to time per-token autoregressive decode instead
    of prefill. Encoder-only models (BERT) auto-skip in decode mode;
    enc-dec (T5) is currently TODO and also skips. Default output basename
    suffixes ``_decode`` when mode=decode so prefill and decode CSVs
    don't collide.
    """
    gpu_sku = gpu_sku.lower()
    if mode not in ("prefill", "decode"):
        raise SystemExit(f"unknown mode: {mode}. Use 'prefill' or 'decode'.")

    model_list = _ALL_MODELS if models == "all" else [m.strip() for m in models.split(",")]
    bs_list = [int(x) for x in batch_sizes.split(",")]
    sl_list = [int(x) for x in seq_lens.split(",")]

    dispatchers = {
        "l4": profile_l4,
        "a10": profile_a10,
        "a100-40gb": profile_a100_40gb,
        "h100": profile_h100,
        "b200": profile_b200,
    }
    if gpu_sku not in dispatchers:
        raise SystemExit(f"unknown gpu-sku: {gpu_sku}. Known: {', '.join(dispatchers)}")

    fn = dispatchers[gpu_sku]
    # CSV name inside the persistent Volume. Use the local output's basename
    # so multiple runs on the same SKU (e.g. split gpt2-large vs other models)
    # land in separate files. Suffix decode CSVs to avoid colliding with
    # prefill data on the same GPU.
    if output:
        out_path = Path(output)
    else:
        suffix = "_decode" if mode == "decode" else ""
        out_path = Path(f"data/raw/{gpu_sku.replace('-', '_')}{suffix}.csv")
    out_basename = out_path.name

    print(f"dispatching to {gpu_sku} (mode={mode}): "
          f"{len(model_list)} models × {len(bs_list)} batches × {len(sl_list)} seqs")
    print(f"  Modal Volume CSV: /csvs/{out_basename}")
    csv_bytes = fn.remote(
        models=model_list, batch_sizes=bs_list, seq_lens=sl_list,
        warmup=warmup, runs=runs, out_basename=out_basename, mode=mode,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(csv_bytes)
    rows = csv_bytes.count(b"\n") - 1
    print(f"wrote {rows} rows ({len(csv_bytes)} bytes) → {out_path}")
