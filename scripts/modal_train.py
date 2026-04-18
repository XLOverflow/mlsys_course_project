"""Dispatch training to Modal H100. Wraps ``scripts/train_and_eval.py``.

Why a separate file from ``modal_profiling.py``: profiling is a per-SKU
concern (each GPU hosts its own profiling session), while training is a
single GPU workload that just needs the fastest card. Keeping the two
apps separate also keeps the profiling volume isolated from training
outputs.

Usage:

  modal run scripts/modal_train.py \
      --split leave-gpu=h100 --epochs 50 \
      --results-out results/table1_leave_h100.json

The train script's stdout is teed to a local log. If ``--results-out``
is given, the Modal side writes a metrics JSON (Method → MAPE/Spearman/
Top-1) that's returned and written locally.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal


# Reuse the same image as profiling — torch≥2.7 covers all GPUs, including B200.
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.7",
        "torch-geometric>=2.4",
        "transformers>=4.35,<4.52",
        "xgboost",
        "numpy", "pandas", "scipy", "scikit-learn", "tqdm",
    )
    # Source + scripts + the two data dirs the training script needs.
    .add_local_dir("src",         remote_path="/root/src")
    .add_local_dir("scripts",     remote_path="/root/scripts")
    .add_local_dir("data/graphs", remote_path="/root/data/graphs")
    .add_local_dir("data/raw",    remote_path="/root/data/raw")
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml")
)

# Separate Volume from the profiling CSVs — this one holds training results
# (metrics JSON, model checkpoints if added later).
RESULTS_VOLUME = modal.Volume.from_name(
    "hetero-cost-training-results", create_if_missing=True,
)

app = modal.App("hetero-cost-training", image=IMAGE)


@app.function(
    gpu="H100!",           # SKU lock — prevent auto-upgrade to H200
    timeout=3600,          # 1 hour per training call
    volumes={"/results": RESULTS_VOLUME},
)
def train_on_h100(
    split: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    backbone: str = "gat",
    constant_h: bool = False,
    few_shot_samples: int = 0,
    filter_noisy: bool = False,
    seed: int = 0,
    out_basename: str = "train_result.json",
) -> dict:
    """Run train_and_eval.py inside the H100 container, return parsed metrics."""
    import os
    import re
    import subprocess
    import sys

    os.chdir("/root")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-e", "/root", "--quiet"]
    )

    log_path = f"/results/{out_basename}.log"
    cmd = [
        sys.executable, "scripts/train_and_eval.py",
        "--csv", "data/raw",
        "--graph-dir", "data/graphs",
        "--split", split,
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--lr", str(lr),
        "--backbone", backbone,
        "--seed", str(seed),
        "--device", "cuda",
    ]
    if constant_h:
        cmd.append("--constant-h")
    if few_shot_samples > 0:
        cmd.extend(["--few-shot-samples", str(few_shot_samples)])
    if filter_noisy:
        cmd.append("--filter-noisy")

    print(">>> " + " ".join(cmd))
    os.makedirs("/results", exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="")
            logf.write(line)
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"train_and_eval.py exited with code {proc.returncode}")

    # Parse the metrics table from log tail. Format comes from train_and_eval.py:
    #   Method              MAPE       Spearman      Top-1
    #   -----
    #   Roofline            22.24%     0.988         0.000
    # We capture the block between the two dashed separators.
    with open(log_path) as f:
        log_text = f.read()
    pattern = re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9 \-\(\)_]+?)\s+"
        r"(?P<mape>[\d.]+)%\s+"
        r"(?P<spearman>-?[\d.]+)\s+"
        r"(?P<top1>[\d.]+)\s*$",
        re.MULTILINE,
    )
    metrics = []
    for m in pattern.finditer(log_text):
        metrics.append({
            "method":   m.group("name").strip(),
            "mape":     float(m.group("mape")),
            "spearman": float(m.group("spearman")),
            "top1":     float(m.group("top1")),
        })

    result = {
        "split": split,
        "epochs": epochs,
        "backbone": backbone,
        "constant_h": constant_h,
        "few_shot_samples": few_shot_samples,
        "filter_noisy": filter_noisy,
        "metrics": metrics,
        "raw_log_path": log_path,   # lives on the Volume
    }

    # Persist the parsed metrics as JSON on the Volume.
    json_path = f"/results/{out_basename}"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    RESULTS_VOLUME.commit()

    return result


@app.local_entrypoint()
def main(
    split: str = "random",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    backbone: str = "gat",
    constant_h: bool = False,
    few_shot_samples: int = 0,
    filter_noisy: bool = False,
    seed: int = 0,
    out: str = "",
):
    """Dispatch a training run to Modal H100 and save results locally."""
    safe_split = split.replace("=", "_").replace("-", "_")
    extras = []
    if constant_h:
        extras.append("constant_h")
    if few_shot_samples > 0:
        extras.append(f"fs{few_shot_samples}")
    if filter_noisy:
        extras.append("filtered")
    suffix = "__" + "_".join(extras) if extras else ""
    default_name = f"{safe_split}_{backbone}_e{epochs}{suffix}.json"
    out_basename = default_name

    print(f"dispatching to H100: split={split} epochs={epochs} backbone={backbone}")
    if constant_h:
        print("  --constant-h (Table 3 row 2 ablation)")
    if few_shot_samples > 0:
        print(f"  --few-shot-samples={few_shot_samples} (Table 2 few-shot)")
    if filter_noisy:
        print("  --filter-noisy")

    result = train_on_h100.remote(
        split=split, epochs=epochs, batch_size=batch_size, lr=lr,
        backbone=backbone, constant_h=constant_h,
        few_shot_samples=few_shot_samples, filter_noisy=filter_noisy,
        seed=seed, out_basename=out_basename,
    )

    out_path = Path(out) if out else Path("results") / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nresult saved → {out_path}")
    print("\n" + "=" * 64)
    print(f"{'Method':<18}  {'MAPE':>8}  {'Spearman':>10}  {'Top-1':>8}")
    print("-" * 64)
    for m in result["metrics"]:
        print(f"{m['method']:<18}  {m['mape']:>7.2f}%  {m['spearman']:>10.3f}  {m['top1']:>8.3f}")
    print("=" * 64)
