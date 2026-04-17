from hetero_cost_model.baselines import roofline_latency
from hetero_cost_model.graph import GraphRepr, NodeFeature
from hetero_cost_model.hardware import HARDWARE_REGISTRY
from hetero_cost_model.strategies import InferenceConfig


def _single_op_graph(flops: float, memory: float) -> GraphRepr:
    nodes = [
        NodeFeature(
            name="n0",
            op_type="linear",
            input_shapes=[(1, 128)],
            output_shape=(1, 128),
            dtype="float32",
            flops=flops,
            memory_bytes=memory,
        )
    ]
    return GraphRepr(nodes=nodes, edges=[])


def test_h100_beats_v100():
    g = _single_op_graph(flops=1e10, memory=1e7)
    cfg = InferenceConfig(batch_size=1, seq_len=128)
    lat_h100 = roofline_latency(g, cfg, HARDWARE_REGISTRY["h100"])
    lat_v100 = roofline_latency(g, cfg, HARDWARE_REGISTRY["v100"])
    assert lat_h100 < lat_v100


def test_larger_batch_increases_latency():
    g = _single_op_graph(flops=1e9, memory=1e6)
    hw = HARDWARE_REGISTRY["h100"]
    lat_bs1 = roofline_latency(g, InferenceConfig(batch_size=1, seq_len=128), hw)
    lat_bs8 = roofline_latency(g, InferenceConfig(batch_size=8, seq_len=128), hw)
    assert lat_bs8 > lat_bs1


def test_b200_beats_h100():
    g = _single_op_graph(flops=1e12, memory=1e8)
    cfg = InferenceConfig(batch_size=4, seq_len=128)
    assert (
        roofline_latency(g, cfg, HARDWARE_REGISTRY["b200"])
        < roofline_latency(g, cfg, HARDWARE_REGISTRY["h100"])
    )
