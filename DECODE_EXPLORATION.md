# Decode Exploration — branch `decode-exploration`

> **Status (2026-04-29)**: timing primitive + run_profiling.py integration
> done; pipeline verified locally on CPU with gpt2-small / gpt2-medium.
> No Modal-side data collected yet, no cost-model training on decode data.
> Branched off main so the poster's prefill story stays clean.

## What's in this branch (vs main)

| File | Change | Status |
|---|---|---|
| [src/hetero_cost_model/profiling.py](src/hetero_cost_model/profiling.py) | Add `time_decode_step()` — single-token forward with populated KV cache | ✅ |
| [scripts/run_profiling.py](scripts/run_profiling.py) | Add `--mode={prefill,decode}` flag + `mode` CSV column + mode-aware resume dedup | ✅ |
| [scripts/decode_smoke.py](scripts/decode_smoke.py) | Standalone smoke test: prints prefill vs decode latency for one model | ✅ |
| [DECODE_EXPLORATION.md](DECODE_EXPLORATION.md) | This document | ✅ |

## Local smoke results (CPU, fp32, small configs)

| Model | Batch | Seq | Prefill p50 (ms) | Decode p50 (ms) | Ratio |
|---|---:|---:|---:|---:|---:|
| gpt2-small | 1 | 32 | 50.3 | 10.1 | 4.96× |
| gpt2-medium | 1 | 32 | 127.6 | 23.2 | 5.51× |

Numbers are **CPU fp32** — not representative of GPU behavior. Useful only
to confirm the pipeline is correct. Actual GPU numbers (especially the
prefill/decode ratio under FP16 on H100) will look very different
(typically prefill is much faster relative to decode because prefill is
compute-bound while decode is memory-bound).

## What works

- **Decoder-only models** (GPT-2 family): full pipeline works.
- **Encoder-only models** (BERT family): cleanly skipped — no autoregressive
  decode exists.
- **CSV schema**: backward-compatible (rows missing `mode` default to
  `"prefill"` on read).
- **Resume**: prefill and decode rows for the same config can coexist;
  re-running skips already-completed rows per `(model, bs, sl, gpu, mode)`.

## What's missing

1. **T5 (encoder-decoder) decode**: deferred. The decode step needs
   `decoder_input_ids` (single new token) plus a `past_key_values` that
   already caches encoder cross-attention. Different forward signature
   from decoder-only models.
2. **GPU profiling runs**: nothing collected on Modal. The 16 Phase-6
   prefill JSONs in `results/` have no decode counterparts.
3. **Graph extractor adaptation**: the existing
   [graph extractor](src/hetero_cost_model/graph/extractor.py) traces
   prefill-shape forwards (`input_ids` of shape `[batch, seq_len]`).
   For decode, the relevant graph is a single-token forward
   (`[batch, 1]`) with populated KV cache — the graph topology is
   similar but FLOPs / memory accounting will differ. Needs separate
   pass or a `mode` flag on extraction.
4. **Sample / data.py extension**: `Sample` does not currently carry
   a `mode` field. Loading mixed prefill/decode CSVs would require
   either two separate datasets or a new field.
5. **Cost-model training on decode data**: no XGBoost / GNN trained
   on decode targets yet. Open question whether the same architecture
   transfers or whether decode needs different features
   (KV-cache-size-aware).

## Modal-side instructions for decode data collection

Once you decide to spend GPU credits, the path is:

```bash
# Pick one GPU first to validate at-scale (B200 sm_100 quirks etc.)
# 1 model × 8 batches × 6 seqs = 48 configs at ~5 min each ≈ 4 hours.
python scripts/run_profiling.py \
    --gpu h100 \
    --models gpt2-small gpt2-medium gpt2-large \
    --batch-sizes 1 2 4 8 16 \
    --seq-lens 32 64 128 256 512 1024 \
    --output data/raw/h100_decode.csv \
    --mode decode

# Then fan out to remaining GPUs (T4/A10/A100/L4/H200/B200 — same
# pattern as prefill collection in two_week_execution_plan.md §3).
```

Estimated wall time per GPU: ~3-6 hours (3 GPT2 models × 30 configs × 100
runs × per-token decode time). Skip BERT (encoder-only) and T5 (TODO).
Total Modal cost ≈ similar order of magnitude to the original prefill
sweep, since per-config wall time is dominated by warmup + 100-run timing
at roughly comparable per-call latency.

## Open design questions for the cost model

1. **Feature space**: prefill features are `(batch, seq_len)`. Decode
   features are `(batch, prompt_len, kv_cache_size, generated_so_far)`.
   For the steady-state per-token measurement on this branch, we hold
   `kv_cache_size = prompt_len` and `generated_so_far = 0`, so the
   feature space collapses to the same `(batch, seq_len)` plus a `mode`
   indicator. That keeps the existing `s` vector usable; cost models
   can take `mode` as a categorical input.
2. **Joint vs separate models**: should one cost model predict both
   prefill and decode (with `mode` as a feature), or two separate
   models? Suggest **separate** for the first experiment — they have
   different roofline regimes (compute-bound vs memory-bound) and
   joint training may dilute both signals.
3. **Router implications**: `predict.py` and the Tier 1/Tier 2 router
   are untouched. If decode prediction becomes a separate model, we
   need a router that ALSO chooses between prefill-mode and decode-mode
   predictors based on the requested phase. Out of scope for this
   branch.

## Recommendation

Do not merge this branch into `main` for the poster build. It contains
infrastructure (timing primitive + CLI flag + smoke script) but no
data, no trained model, no claim. Keep it warm for follow-up work
after the deadline.

If you want a single line on the poster:

> "Decode-stage prediction infrastructure prototyped on a separate
> branch (timing primitive + CLI) but data collection deferred to
> future work — Forward-pass / prefill latency is the cost-model
> protocol established here; KV-cache-aware decode prediction is the
> natural extension."
