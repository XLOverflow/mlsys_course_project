from hetero_cost_model.baselines import roofline_latency
from hetero_cost_model.graph import GraphRepr, NodeFeature
from hetero_cost_model.hardware import HARDWARE_REGISTRY
from hetero_cost_model.strategies import Strategy


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


def test_roofline_gpu_beats_cpu():
    g = _single_op_graph(flops=1e9, memory=1e6)
    hw = HARDWARE_REGISTRY["h100"]
    lat_gpu = roofline_latency(g, Strategy(placements=[1]), hw)
    lat_cpu = roofline_latency(g, Strategy(placements=[0]), hw)
    assert lat_gpu < lat_cpu


def test_h100_beats_v100_on_gpu():
    g = _single_op_graph(flops=1e10, memory=1e7)
    s = Strategy(placements=[1])
    assert (
        roofline_latency(g, s, HARDWARE_REGISTRY["h100"])
        < roofline_latency(g, s, HARDWARE_REGISTRY["v100"])
    )
