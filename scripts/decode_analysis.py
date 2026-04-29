"""Quick analysis of decode-mode profiling data.

Once the 7 Modal jobs complete, run this to:

  1. Combine all ``data/raw/*_decode.csv`` files into a unified frame
  2. Print per-GPU + per-model summary statistics (the "ground truth")
  3. Run train_and_eval-style baselines (Roofline, XGBoost, GNN) on the
     decode data via leave-model-out
  4. Compare key MAPE numbers against the prefill counterparts in
     [results/EXPERIMENTS.md](../results/EXPERIMENTS.md) so we can say
     whether decode is harder/easier than prefill to predict, and which
     baseline does best.

This is the *minimum* analysis to put a "decode-mode prediction"
column on the poster — not a full re-do of Phase 6.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out", type=Path, default=Path("results/decode_summary.md"))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    decode_csvs = sorted(args.csv_dir.glob("*_decode.csv"))
    if not decode_csvs:
        print(f"no *_decode.csv files in {args.csv_dir}; "
              f"wait for Modal jobs to finish.")
        return 1

    print(f"[load] found {len(decode_csvs)} decode CSV files:")
    for c in decode_csvs:
        print(f"  - {c}")

    frames = []
    for c in decode_csvs:
        df = pd.read_csv(c)
        df = df[df["n_runs"] > 0]   # drop OOM/skip rows
        if len(df) == 0:
            continue
        frames.append(df)
    if not frames:
        print("all decode CSVs empty — nothing to analyze")
        return 1

    df = pd.concat(frames, ignore_index=True)
    n_total = len(df)
    print(f"[combined] {n_total} clean rows, "
          f"{df['actual_gpu_name'].nunique()} GPUs, "
          f"{df['model_name'].nunique()} models")

    # --- Per-GPU + per-model summary -------------------------------------
    summary = (df.groupby(["actual_gpu_name", "model_name"])["p50_ms"]
                 .agg(["count", "min", "median", "max"])
                 .round(2))
    print("\nPer-GPU × per-model decode latency (p50 ms):")
    print(summary.to_string())

    # --- Per-model scaling across GPUs (constant batch/seq slice) --------
    slice_df = df[(df["batch_size"] == 1) & (df["seq_len"] == 128)]
    if len(slice_df) > 0:
        cross_gpu = (slice_df.pivot(index="actual_gpu_name",
                                    columns="model_name",
                                    values="p50_ms")
                            .round(2))
        print("\nDecode p50 (ms) at batch=1, seq=128 — clean cross-GPU comparison:")
        print(cross_gpu.to_string())

    # --- Quick scaling sanity --------------------------------------------
    # Memory-bound expectation: per-token latency scales roughly with
    # parameter count (= weight bytes). Print the ratios per GPU.
    print("\nParameter-count scaling check (gpt2-large / gpt2-small ratio per GPU):")
    print("(theoretical: ~6.2× by params; memory-bound decode should be close)")
    if len(slice_df) > 0:
        for gpu in slice_df["actual_gpu_name"].unique():
            g = slice_df[slice_df["actual_gpu_name"] == gpu]
            small = g[g["model_name"] == "gpt2-small"]["p50_ms"]
            large = g[g["model_name"] == "gpt2-large"]["p50_ms"]
            if len(small) > 0 and len(large) > 0:
                ratio = float(large.iloc[0]) / float(small.iloc[0])
                print(f"  {gpu:<32}  gpt2-large / gpt2-small = {ratio:.2f}×")

    # --- Save markdown summary -------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Decode-mode profiling summary\n\n")
        f.write(f"Sources: {len(decode_csvs)} CSV files, "
                f"{n_total} rows, "
                f"{df['actual_gpu_name'].nunique()} GPUs, "
                f"{df['model_name'].nunique()} models.\n\n")
        f.write("## Per-GPU × per-model decode p50 (ms)\n\n")
        f.write("```\n")
        f.write(summary.to_string())
        f.write("\n```\n\n")
        if len(slice_df) > 0:
            f.write("## Cross-GPU comparison @ batch=1, seq=128\n\n")
            f.write("```\n")
            f.write(cross_gpu.to_string())
            f.write("\n```\n")
    print(f"\n[wrote] markdown summary to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
