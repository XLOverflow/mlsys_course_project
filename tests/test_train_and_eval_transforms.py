"""Unit tests for pre-training data transforms used by train_and_eval.py:

  - apply_constant_h   (hardware ablation for Table 3 row 2)
  - apply_few_shot     (H200/B200 few-shot for Table 2)
  - filter_noisy       (optional --filter-noisy flag)
  - load_samples       (multi-CSV merge with dedup)
"""
import pickle
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

# train_and_eval.py lives in scripts/; import it via path append to avoid
# requiring the tests to invoke it as a subprocess.
_SCRIPTS = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import train_and_eval as te  # noqa: E402

from hetero_cost_model.data import Sample
from hetero_cost_model.graph import GraphRepr
from hetero_cost_model.hardware import HARDWARE_REGISTRY, Hardware
from hetero_cost_model.strategies import InferenceConfig

GRAPH_DIR = Path("data/graphs")


# --- Helpers -----------------------------------------------------------------

def _build_samples(gpus=("v100", "a100", "h100"), latency=10.0) -> list:
    """Small deterministic Sample list for transform tests."""
    with open(GRAPH_DIR / "bert-base.pkl", "rb") as f:
        g: GraphRepr = pickle.load(f)
    return [
        Sample(
            graph=g,
            config=InferenceConfig(batch_size=bs, seq_len=64),
            hardware=HARDWARE_REGISTRY[gk],
            latency_ms=latency + 0.1 * bs,
            model_name="bert-base",
        )
        for gk in gpus for bs in (1, 2, 4)
    ]


# --- apply_constant_h --------------------------------------------------------

def test_mean_hardware_averages_every_dim():
    """Each continuous dim of the mean Hardware must equal the mean of that
    dim across the input samples. SM count rounds to the nearest int (since
    ``Hardware.sm_count: int``) so its tolerance is within 1."""
    samples = _build_samples()
    mean_h = te._mean_hardware(samples)

    import numpy as np
    raw = np.array([s.hardware.to_vector(normalize=False) for s in samples])
    expected = raw.mean(axis=0)
    got = mean_h.to_vector(normalize=False)
    # dims 0..3 are continuous floats (TFLOPS / mem_gb / bw / L2); dim 4 is sm_count (int)
    for i in range(4):
        assert abs(expected[i] - got[i]) < 1e-6, (
            f"dim {i}: expected {expected[i]}, got {got[i]}"
        )
    assert abs(expected[4] - got[4]) <= 0.5, (
        f"sm_count dim: expected ≈{expected[4]}, got {got[4]} (should round)"
    )


def test_apply_constant_h_replaces_every_hardware_in_place_of_copy():
    """Returned samples all share the same Hardware instance, which is the
    provided mean_h; no sample's hardware is the original Volta/Ampere one."""
    samples = _build_samples()
    mean_h = te._mean_hardware(samples)
    rewritten = te.apply_constant_h(samples, mean_h)

    assert len(rewritten) == len(samples)
    for s_new, s_old in zip(rewritten, samples):
        assert s_new.hardware is mean_h
        # Non-hardware fields must survive unchanged
        assert s_new.graph is s_old.graph
        assert s_new.config == s_old.config
        assert s_new.latency_ms == s_old.latency_ms
        assert s_new.model_name == s_old.model_name


def test_constant_h_never_uses_real_gpu_name():
    """Guards against a refactor that accidentally reuses a real Hardware."""
    mean_h = te._mean_hardware(_build_samples())
    real_names = {hw.name for hw in HARDWARE_REGISTRY.values()}
    assert mean_h.name not in real_names


# --- apply_few_shot ----------------------------------------------------------

def test_few_shot_zero_is_identity():
    tr = _build_samples(gpus=("a100",), latency=10.0)
    te_ = _build_samples(gpus=("b200",), latency=20.0)
    tr2, te2 = te.apply_few_shot(tr, te_, n=0, seed=0)
    assert tr2 == tr and te2 == te_


def test_few_shot_moves_n_samples_from_test_to_train():
    tr = _build_samples(gpus=("a100",), latency=10.0)
    te_ = _build_samples(gpus=("b200",), latency=20.0)
    n_tr = len(tr)
    n_te = len(te_)

    tr2, te2 = te.apply_few_shot(tr, te_, n=2, seed=0)
    assert len(tr2) == n_tr + 2
    assert len(te2) == n_te - 2
    # The moved samples should originate from the test set (b200 hardware),
    # not from the training set (a100).
    moved = tr2[n_tr:]
    assert all(s.hardware.name == "B200" for s in moved)


def test_few_shot_deterministic_for_same_seed():
    """Need a test pool bigger than n so few-shot doesn't drain it."""
    tr = _build_samples(gpus=("a100",))
    # b200 pool needs > 3 samples so n=3 few-shot leaves a non-empty test set.
    te_ = _build_samples(gpus=("b200", "l4"))     # 6 samples
    tr_a, te_a = te.apply_few_shot(tr, te_, n=3, seed=42)
    tr_b, te_b = te.apply_few_shot(tr, te_, n=3, seed=42)
    assert [s.config for s in tr_a] == [s.config for s in tr_b]
    assert [s.config for s in te_a] == [s.config for s in te_b]


def test_few_shot_raises_when_n_drains_test():
    tr = _build_samples(gpus=("a100",))
    te_ = _build_samples(gpus=("b200",))
    with pytest.raises(ValueError):
        te.apply_few_shot(tr, te_, n=len(te_), seed=0)


# --- filter_noisy ------------------------------------------------------------

def test_filter_noisy_drops_only_flagged_rows():
    samples = _build_samples(gpus=("a100",))
    # Build a lookup that flags exactly the (bs=2, sl=64, a100) row as noisy.
    flagged_key = ("bert-base", 2, 64, "A100")
    lookup = {flagged_key: True}
    filtered = te.filter_noisy(samples, lookup)
    assert len(filtered) == len(samples) - 1
    for s in filtered:
        key = (s.model_name, s.config.batch_size, s.config.seq_len, s.hardware.name)
        assert key != flagged_key


def test_filter_noisy_is_noop_when_all_clean():
    samples = _build_samples()
    filtered = te.filter_noisy(samples, {})   # nothing flagged
    assert len(filtered) == len(samples)


# --- load_samples (multi-CSV + dedup) ----------------------------------------

def test_load_samples_deduplicates_across_overlapping_files(tmp_path):
    """If the same (model, bs, sl, actual_gpu_name) row appears in two CSVs,
    the loader should keep only the first occurrence."""
    # Use an existing committed CSV to avoid rebuilding a synthetic one.
    real_csv = Path("data/raw/a100.csv")
    if not real_csv.exists():
        pytest.skip("a100.csv not present in this checkout")

    # Load once.
    baseline = te.load_samples([real_csv], GRAPH_DIR)

    # Load twice (same file). Dedup should leave the count unchanged.
    doubled = te.load_samples([real_csv, real_csv], GRAPH_DIR)
    assert len(doubled) == len(baseline)


def test_load_samples_accepts_directory():
    """Passing data/raw/ should pick up all CSVs inside."""
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw/ not present")
    samples = te.load_samples([raw_dir], GRAPH_DIR)
    assert len(samples) > 0
    # Should include multiple hardware names.
    names = {s.hardware.name for s in samples}
    assert len(names) >= 2


def test_load_samples_rejects_nonexistent_path(tmp_path):
    with pytest.raises(SystemExit):
        te.load_samples([tmp_path / "does_not_exist.csv"], GRAPH_DIR)
