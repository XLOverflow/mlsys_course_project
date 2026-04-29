"""Smoke-test the decode-mode timing primitive on a small model.

Runs locally on CPU (slow but functional) to verify the
``time_decode_step`` plumbing works end-to-end before deploying to
Modal GPUs. Intentionally minimal — no CSV writing, no config sweep —
just one model + one config + a printed comparison of prefill latency
vs single-token decode latency.

Encoder-only models (BERT family) are skipped; they have no
autoregressive decode.

Usage:

    # CPU smoke test — small batch/seq so it finishes in seconds
    python scripts/decode_smoke.py --model gpt2-small --batch 1 --seq 32

    # On a GPU machine (Modal), bump warmup/runs and use real configs:
    python scripts/decode_smoke.py --model gpt2-medium --batch 4 --seq 256 \\
        --device cuda:0 --warmup 50 --runs 100
"""
from __future__ import annotations

import argparse
import sys
from typing import Tuple

import torch

from hetero_cost_model.model_zoo import MODEL_BY_NAME, ModelSpec, load_model
from hetero_cost_model.profiling import profile_callable, time_decode_step


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gpt2-small",
                   help="Model name from model_zoo (must be decoder or enc-dec)")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=32, help="Prompt length (tokens)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--warmup", type=int, default=3,
                   help="Default 3 for CPU smoke test; bump to 50 on GPU")
    p.add_argument("--runs", type=int, default=10,
                   help="Default 10 for CPU smoke test; bump to 100 on GPU")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32",
                   help="fp32 on CPU (no fp16 support); fp16 on GPU")
    return p.parse_args()


def _resolve_spec(name: str) -> ModelSpec:
    if name not in MODEL_BY_NAME:
        raise SystemExit(f"unknown model: {name}. Available: {sorted(MODEL_BY_NAME)}")
    spec = MODEL_BY_NAME[name]
    if spec.family == "encoder":
        raise SystemExit(
            f"model '{name}' is encoder-only — it has no autoregressive "
            f"decode step. Use a decoder model (gpt2-*)."
        )
    if spec.family == "enc-dec":
        # TODO: T5 decode step requires distinct plumbing — single new
        # decoder_input_ids + past_key_values that already caches
        # encoder cross-attention. Left out of this exploratory smoke
        # script; the GPT2 family (decoder-only) is the cleanest
        # validation target.
        raise SystemExit(
            f"model '{name}' is encoder-decoder; decode-step timing is "
            f"left as TODO for this exploratory branch. "
            f"Use a decoder-only model (gpt2-small/medium/large)."
        )
    return spec


def _make_inputs(spec: ModelSpec, batch: int, seq: int, device: str) -> Tuple[torch.Tensor, dict]:
    """Build (input_ids, extra_kwargs) for the chosen model family."""
    input_ids = torch.randint(0, 1000, (batch, seq), dtype=torch.long, device=device)
    extra: dict = {}
    if spec.family == "enc-dec":
        # T5 needs decoder_input_ids for the encoder pass; the decode step
        # itself uses the cached past_key_values rather than these.
        extra["decoder_input_ids"] = torch.randint(
            0, 1000, (batch, seq), dtype=torch.long, device=device,
        )
    return input_ids, extra


def main() -> int:
    args = parse_args()
    spec = _resolve_spec(args.model)

    print(f"[setup]   model={spec.name}  family={spec.family}  "
          f"batch={args.batch}  seq={args.seq}  device={args.device}  dtype={args.dtype}")

    print("[load]    loading model ...")
    model = load_model(spec).to(args.device)
    if args.dtype == "fp16":
        model = model.half()
    model = model.eval()

    input_ids, extra = _make_inputs(spec, args.batch, args.seq, args.device)

    # --- Prefill timing (existing protocol) -------------------------------
    print("[prefill] timing full forward over the prompt ...")

    def prefill_step() -> None:
        with torch.inference_mode():
            model(input_ids, **extra)

    prefill = profile_callable(
        prefill_step, warmup=args.warmup, runs=args.runs, device=args.device,
    )
    print(f"[prefill] p50={prefill.p50_ms:8.3f} ms  mean={prefill.mean_ms:8.3f} ms  "
          f"std={prefill.std_ms:.3f}  noisy={prefill.noisy}")

    # --- Decode timing (new primitive) -----------------------------------
    print("[decode]  timing single-token forward with populated KV cache ...")
    decode = time_decode_step(
        model, input_ids, extra=extra,
        warmup=args.warmup, runs=args.runs, device=args.device,
    )
    print(f"[decode]  p50={decode.p50_ms:8.3f} ms  mean={decode.mean_ms:8.3f} ms  "
          f"std={decode.std_ms:.3f}  noisy={decode.noisy}")

    # --- Comparison summary -----------------------------------------------
    ratio = prefill.p50_ms / decode.p50_ms if decode.p50_ms > 0 else float("nan")
    per_prompt_token_prefill = prefill.p50_ms / args.seq
    print()
    print("─" * 64)
    print(f"  prefill / decode ratio          : {ratio:6.2f}×")
    print(f"  prefill per-token (= prefill/seq): {per_prompt_token_prefill:6.3f} ms/token")
    print(f"  decode  per-token               : {decode.p50_ms:6.3f} ms/token")
    print("─" * 64)
    print("\nNote on interpretation:")
    print("  - On compute-bound prefill: per-token cost is small because")
    print("    tokens are processed in parallel.")
    print("  - On memory-bound decode: per-token cost is dominated by")
    print("    weight + KV cache reads; usually higher than prefill/seq.")
    print("  - On CPU these numbers are not representative of GPU behavior;")
    print("    use this script to verify the pipeline, then re-run on Modal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
