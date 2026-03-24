import torch

from vgl import Graph
from vgl.data.ondisk import deserialize_graph
from vgl.distributed.partition import load_partition_manifest
from vgl.distributed.writer import write_partitioned_graph


def test_partition_writer_splits_graph_into_local_files(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )

    manifest = write_partitioned_graph(graph, tmp_path, num_partitions=2)
    loaded = load_partition_manifest(tmp_path / "manifest.json")
    part0 = torch.load(tmp_path / "part-0.pt", weights_only=True)
    part1 = torch.load(tmp_path / "part-1.pt", weights_only=True)

    assert manifest.num_partitions == 2
    assert loaded.owner(0).partition_id == 0
    assert loaded.owner(3).partition_id == 1
    part0_graph = deserialize_graph(part0["graph"])
    part1_graph = deserialize_graph(part1["graph"])

    assert torch.equal(part0["node_ids"], torch.tensor([0, 1]))
    assert torch.equal(part1["node_ids"], torch.tensor([2, 3]))
    assert torch.equal(part0_graph.edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(part1_graph.edge_index, torch.tensor([[0], [1]]))


def test_partition_writer_preserves_temporal_graph_payloads(tmp_path):
    graph = Graph.temporal(
        nodes={"node": {"x": torch.arange(8, dtype=torch.float32).view(4, 2)}},
        edges={
            ("node", "to", "node"): {
                "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
                "timestamp": torch.tensor([3, 5, 7, 11]),
            }
        },
        time_attr="timestamp",
    )

    manifest = write_partitioned_graph(graph, tmp_path, num_partitions=2)
    part0 = torch.load(tmp_path / "part-0.pt", weights_only=True)
    part1 = torch.load(tmp_path / "part-1.pt", weights_only=True)

    assert manifest.num_partitions == 2
    assert torch.equal(part0["node_ids"], torch.tensor([0, 1]))
    assert torch.equal(part1["node_ids"], torch.tensor([2, 3]))

    part0_graph = deserialize_graph(part0["graph"])
    part1_graph = deserialize_graph(part1["graph"])

    assert part0_graph.schema.time_attr == "timestamp"
    assert part1_graph.schema.time_attr == "timestamp"
    assert torch.equal(part0_graph.edata["timestamp"], torch.tensor([3]))
    assert torch.equal(part1_graph.edata["timestamp"], torch.tensor([7]))


def test_partition_writer_preserves_multi_relation_graph_payloads(tmp_path):
    follows = ("node", "follows", "node")
    likes = ("node", "likes", "node")
    graph = Graph.hetero(
        nodes={"node": {"x": torch.arange(8, dtype=torch.float32).view(4, 2)}},
        edges={
            follows: {
                "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            },
            likes: {
                "edge_index": torch.tensor([[1, 0, 3], [0, 1, 2]]),
                "score": torch.tensor([0.5, 0.6, 0.7]),
            },
        },
    )

    manifest = write_partitioned_graph(graph, tmp_path, num_partitions=2)
    part0 = torch.load(tmp_path / "part-0.pt", weights_only=True)
    part1 = torch.load(tmp_path / "part-1.pt", weights_only=True)

    assert manifest.num_partitions == 2

    part0_graph = deserialize_graph(part0["graph"])
    part1_graph = deserialize_graph(part1["graph"])

    assert set(part0_graph.edges) == {follows, likes}
    assert set(part1_graph.edges) == {follows, likes}
    assert torch.equal(part0_graph.edges[follows].edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(part0_graph.edges[follows].weight, torch.tensor([1.0]))
    assert torch.equal(part0_graph.edges[likes].edge_index, torch.tensor([[1, 0], [0, 1]]))
    assert torch.equal(part0_graph.edges[likes].score, torch.tensor([0.5, 0.6]))
    assert torch.equal(part1_graph.edges[follows].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(part1_graph.edges[follows].weight, torch.tensor([3.0, 4.0]))
    assert torch.equal(part1_graph.edges[likes].edge_index, torch.tensor([[1], [0]]))
    assert torch.equal(part1_graph.edges[likes].score, torch.tensor([0.7]))
