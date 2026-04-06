import torch

from vgl import Graph
from vgl.nn import IdentityTemporalMessage
from vgl.nn import LastMessageAggregator
from vgl.nn import MeanMessageAggregator
from vgl.nn import TGNMemory
from vgl.nn import TGATEncoder
from vgl.nn import TGATLayer
from vgl.nn import TimeEncoder


def _temporal_graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )


def test_time_encoder_accepts_vector_input():
    encoder = TimeEncoder(out_channels=4)

    out = encoder(torch.tensor([1.0, 3.0]))

    assert out.shape == (2, 4)


def test_identity_temporal_message_concatenates_memory_message_and_time():
    module = IdentityTemporalMessage(memory_channels=3, raw_message_channels=2, time_channels=4)

    out = module(
        src_memory=torch.randn(2, 3),
        dst_memory=torch.randn(2, 3),
        raw_message=torch.randn(2, 2),
        delta_time=torch.tensor([1.0, 2.0]),
    )

    assert out.shape == (2, 12)


def test_last_message_aggregator_picks_latest_message_per_node():
    aggregator = LastMessageAggregator()

    node_ids, messages, timestamps = aggregator(
        messages=torch.tensor([[1.0, 0.0], [2.0, 0.0], [5.0, 1.0]]),
        node_index=torch.tensor([0, 0, 1]),
        timestamp=torch.tensor([1, 3, 2]),
        num_nodes=3,
    )

    assert torch.equal(node_ids, torch.tensor([0, 1]))
    assert torch.equal(messages, torch.tensor([[2.0, 0.0], [5.0, 1.0]]))
    assert torch.equal(timestamps, torch.tensor([3, 2]))


def test_mean_message_aggregator_averages_messages_per_node():
    aggregator = MeanMessageAggregator()

    node_ids, messages, timestamps = aggregator(
        messages=torch.tensor([[1.0, 1.0], [3.0, 5.0], [4.0, 2.0]]),
        node_index=torch.tensor([0, 0, 2]),
        timestamp=torch.tensor([1, 4, 3]),
        num_nodes=4,
    )

    assert torch.equal(node_ids, torch.tensor([0, 2]))
    assert torch.allclose(messages, torch.tensor([[2.0, 3.0], [4.0, 2.0]]))
    assert torch.equal(timestamps, torch.tensor([4, 3]))


def test_last_message_aggregator_avoids_tensor_tolist(monkeypatch):
    aggregator = LastMessageAggregator()

    def fail_tolist(self):
        raise AssertionError("LastMessageAggregator should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    node_ids, messages, timestamps = aggregator(
        messages=torch.tensor([[1.0, 0.0], [2.0, 0.0], [5.0, 1.0]]),
        node_index=torch.tensor([0, 0, 1]),
        timestamp=torch.tensor([1, 3, 2]),
        num_nodes=3,
    )

    assert torch.equal(node_ids, torch.tensor([0, 1]))
    assert torch.equal(messages, torch.tensor([[2.0, 0.0], [5.0, 1.0]]))
    assert torch.equal(timestamps, torch.tensor([3, 2]))


def test_mean_message_aggregator_avoids_tensor_tolist(monkeypatch):
    aggregator = MeanMessageAggregator()

    def fail_tolist(self):
        raise AssertionError("MeanMessageAggregator should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    node_ids, messages, timestamps = aggregator(
        messages=torch.tensor([[1.0, 1.0], [3.0, 5.0], [4.0, 2.0]]),
        node_index=torch.tensor([0, 0, 2]),
        timestamp=torch.tensor([1, 4, 3]),
        num_nodes=4,
    )

    assert torch.equal(node_ids, torch.tensor([0, 2]))
    assert torch.allclose(messages, torch.tensor([[2.0, 3.0], [4.0, 2.0]]))
    assert torch.equal(timestamps, torch.tensor([4, 3]))


def test_last_message_aggregator_avoids_torch_unique(monkeypatch):
    aggregator = LastMessageAggregator()

    def fail_unique(*args, **kwargs):
        raise AssertionError("LastMessageAggregator should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    node_ids, messages, timestamps = aggregator(
        messages=torch.tensor([[1.0, 0.0], [2.0, 0.0], [5.0, 1.0]]),
        node_index=torch.tensor([0, 0, 1]),
        timestamp=torch.tensor([1, 3, 2]),
        num_nodes=3,
    )

    assert torch.equal(node_ids, torch.tensor([0, 1]))
    assert torch.equal(messages, torch.tensor([[2.0, 0.0], [5.0, 1.0]]))
    assert torch.equal(timestamps, torch.tensor([3, 2]))


def test_mean_message_aggregator_avoids_torch_unique(monkeypatch):
    aggregator = MeanMessageAggregator()

    def fail_unique(*args, **kwargs):
        raise AssertionError("MeanMessageAggregator should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    node_ids, messages, timestamps = aggregator(
        messages=torch.tensor([[1.0, 1.0], [3.0, 5.0], [4.0, 2.0]]),
        node_index=torch.tensor([0, 0, 2]),
        timestamp=torch.tensor([1, 4, 3]),
        num_nodes=4,
    )

    assert torch.equal(node_ids, torch.tensor([0, 2]))
    assert torch.allclose(messages, torch.tensor([[2.0, 3.0], [4.0, 2.0]]))
    assert torch.equal(timestamps, torch.tensor([4, 3]))


def test_tgat_layer_accepts_graph_input():
    encoder = TGATLayer(in_channels=4, out_channels=4, time_channels=4, heads=2, dropout=0.0)

    out = encoder(_temporal_graph(), query_time=torch.tensor(5.0))

    assert out.shape == (3, 4)


def test_tgat_encoder_accepts_tensor_inputs():
    graph = _temporal_graph()
    encoder = TGATEncoder(channels=4, num_layers=2, time_channels=4, heads=2, dropout=0.0)

    out = encoder(
        graph.x,
        graph.edge_index,
        edge_time=graph.edata["timestamp"],
        query_time=torch.tensor(5.0),
    )

    assert out.shape == (3, 4)


def test_tgn_memory_updates_only_touched_nodes_and_tracks_last_update():
    memory = TGNMemory(
        num_nodes=4,
        memory_channels=5,
        raw_message_channels=2,
        time_channels=4,
    )

    memory.update(
        src_index=torch.tensor([0, 1]),
        dst_index=torch.tensor([1, 2]),
        timestamp=torch.tensor([2, 5]),
        raw_message=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    )

    state = memory()

    assert state.shape == (4, 5)
    assert torch.any(state[0] != 0)
    assert torch.any(state[1] != 0)
    assert torch.any(state[2] != 0)
    assert torch.all(state[3] == 0)
    assert torch.equal(memory.last_update, torch.tensor([2, 5, 5, 0]))


def test_tgn_memory_reset_state_clears_memory_and_timestamps():
    memory = TGNMemory(
        num_nodes=3,
        memory_channels=4,
        raw_message_channels=2,
        time_channels=4,
    )
    memory.update(
        src_index=torch.tensor([0]),
        dst_index=torch.tensor([1]),
        timestamp=torch.tensor([3]),
        raw_message=torch.tensor([[0.5, 0.2]]),
    )

    memory.reset_state()

    assert torch.all(memory() == 0)
    assert torch.equal(memory.last_update, torch.zeros(3, dtype=torch.long))
