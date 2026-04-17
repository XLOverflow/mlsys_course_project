from hetero_cost_model.strategies import (
    CONFIG_FEATURE_DIM,
    InferenceConfig,
    config_grid,
    config_pairs,
)


def test_config_vector_normalized():
    c = InferenceConfig(batch_size=4, seq_len=128)
    vec = c.to_vector()
    assert len(vec) == CONFIG_FEATURE_DIM
    assert all(0.0 <= v <= 1.0 for v in vec)


def test_config_vector_scales_with_batch():
    small = InferenceConfig(batch_size=1, seq_len=128).to_vector()
    large = InferenceConfig(batch_size=8, seq_len=128).to_vector()
    assert large[0] > small[0]


def test_config_grid_cartesian_product():
    grid = config_grid(batch_sizes=(1, 4), seq_lens=(64, 128, 256))
    assert len(grid) == 2 * 3
    assert all(isinstance(c, InferenceConfig) for c in grid)


def test_config_grid_default():
    grid = config_grid()
    assert len(grid) == 3 * 3  # (1,4,8) x (64,128,256)


def test_config_pairs_count():
    grid = config_grid(batch_sizes=(1, 4), seq_lens=(64, 128))
    pairs = config_pairs(grid)
    n = len(grid)
    assert len(pairs) == n * (n - 1) // 2
