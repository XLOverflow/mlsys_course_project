from hetero_cost_model.graph import GraphRepr, NodeFeature
from hetero_cost_model.strategies import (
    enumerate_strategies,
    full_device,
    identify_strategic_nodes,
    random_strategies,
)


def _fake_graph(n: int = 8) -> GraphRepr:
    nodes = [
        NodeFeature(
            name=f"n{i}",
            op_type="linear",
            input_shapes=[(1, 4)],
            output_shape=(1, 4),
            dtype="float32",
            flops=float(i + 1) * 10.0,
            memory_bytes=100.0,
        )
        for i in range(n)
    ]
    edges = [(i, i + 1) for i in range(n - 1)]
    return GraphRepr(nodes=nodes, edges=edges, name="fake")


def test_identify_strategic_picks_highest_flops():
    g = _fake_graph(8)
    hot = identify_strategic_nodes(g, k=3)
    assert hot[0] == 7
    assert set(hot) == {5, 6, 7}


def test_enumerate_size_matches_combinations():
    g = _fake_graph(8)
    strats = enumerate_strategies(g, strategic_k=3)
    assert len(strats) == 2 ** 3
    assert all(len(s.placements) == 8 for s in strats)


def test_enumerate_with_batch_grid():
    g = _fake_graph(6)
    strats = enumerate_strategies(g, strategic_k=2, batch_sizes=(1, 4))
    assert len(strats) == 2 ** 2 * 2


def test_random_and_full_device():
    g = _fake_graph(5)
    assert len(random_strategies(g, 4, seed=1)) == 4
    assert full_device(g, device=1).num_gpu_ops() == 5
