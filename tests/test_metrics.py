from hetero_cost_model.metrics import mape, ndcg, spearman, top_k_accuracy


def test_mape_perfect_prediction():
    assert mape([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mape_ten_percent_error():
    assert abs(mape([1.1], [1.0]) - 0.1) < 1e-9


def test_spearman_monotonic():
    assert abs(spearman([1, 2, 3], [10, 20, 30]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3], [30, 20, 10]) + 1.0) < 1e-9


def test_top_k_recovers_fast_items():
    assert top_k_accuracy([0.1, 0.2, 0.3, 0.4], [1.0, 2.0, 3.0, 4.0], k=2) == 1.0


def test_ndcg_perfect_ordering_is_one():
    assert abs(ndcg([1, 2, 3], [10, 20, 30], k=3) - 1.0) < 1e-9
