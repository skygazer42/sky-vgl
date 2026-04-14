import torch

from vgl import Graph
from vgl.data.ondisk import GRAPH_PAYLOAD_FORMAT, GRAPH_PAYLOAD_FORMAT_VERSION, deserialize_graph
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
    assert part0["graph"]["format"] == GRAPH_PAYLOAD_FORMAT
    assert part0["graph"]["format_version"] == GRAPH_PAYLOAD_FORMAT_VERSION
    assert part1["graph"]["format"] == GRAPH_PAYLOAD_FORMAT
    assert part1["graph"]["format_version"] == GRAPH_PAYLOAD_FORMAT_VERSION
    part0_graph = deserialize_graph(part0["graph"])
    part1_graph = deserialize_graph(part1["graph"])

    assert torch.equal(part0["node_ids"], torch.tensor([0, 1]))
    assert torch.equal(part1["node_ids"], torch.tensor([2, 3]))
    assert torch.equal(part0_graph.edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(part1_graph.edge_index, torch.tensor([[0], [1]]))


def test_partition_writer_avoids_tensor_tolist(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )

    def fail_tolist(self):
        raise AssertionError("write_partitioned_graph should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    manifest = write_partitioned_graph(graph, tmp_path, num_partitions=2)

    assert manifest.num_partitions == 2


def test_partition_writer_avoids_tensor_item(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )

    def fail_item(self):
        raise AssertionError("write_partitioned_graph should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    manifest = write_partitioned_graph(graph, tmp_path, num_partitions=2)

    assert manifest.num_partitions == 2


def test_partition_writer_avoids_tensor_int(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )

    def fail_int(self):
        raise AssertionError("write_partitioned_graph should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    manifest = write_partitioned_graph(graph, tmp_path, num_partitions=2)

    assert manifest.num_partitions == 2


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


def test_partition_writer_preserves_heterogeneous_partition_payloads(tmp_path):
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.arange(8, dtype=torch.float32).view(4, 2)},
            "paper": {"x": torch.arange(10, 18, dtype=torch.float32).view(4, 2)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]]),
                "score": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9]),
            },
        },
    )

    manifest = write_partitioned_graph(graph, tmp_path, num_partitions=2)
    loaded = load_partition_manifest(tmp_path / "manifest.json")
    part0 = torch.load(tmp_path / "part-0.pt", weights_only=True)
    part1 = torch.load(tmp_path / "part-1.pt", weights_only=True)

    assert manifest.num_nodes_by_type == {"author": 4, "paper": 4}
    assert loaded.owner(3, node_type="author").partition_id == 1
    assert loaded.owner(3, node_type="paper").partition_id == 1
    assert torch.equal(part0["node_ids"]["author"], torch.tensor([0, 1]))
    assert torch.equal(part0["node_ids"]["paper"], torch.tensor([0, 1]))
    assert torch.equal(part1["node_ids"]["author"], torch.tensor([2, 3]))
    assert torch.equal(part1["node_ids"]["paper"], torch.tensor([2, 3]))

    part0_graph = deserialize_graph(part0["graph"])
    part1_graph = deserialize_graph(part1["graph"])

    assert set(part0_graph.nodes) == {"author", "paper"}
    assert set(part0_graph.edges) == {writes, cites}
    assert torch.equal(part0_graph.nodes["author"].x, torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    assert torch.equal(part1_graph.nodes["paper"].x, torch.tensor([[14.0, 15.0], [16.0, 17.0]]))
    assert torch.equal(part0_graph.edges[writes].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(part0_graph.edges[writes].weight, torch.tensor([1.0, 2.0]))
    assert torch.equal(part1_graph.edges[writes].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(part1_graph.edges[writes].weight, torch.tensor([3.0, 4.0]))
    assert torch.equal(part0_graph.edges[cites].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(part0_graph.edges[cites].score, torch.tensor([0.1, 0.2]))
    assert torch.equal(part1_graph.edges[cites].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(part1_graph.edges[cites].score, torch.tensor([0.3, 0.4]))



def test_partition_writer_persists_boundary_edges_in_global_id_space(tmp_path):
    edge_type = ("node", "to", "node")
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )

    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    part0 = torch.load(tmp_path / "part-0.pt", weights_only=True)
    part1 = torch.load(tmp_path / "part-1.pt", weights_only=True)

    boundary0 = part0["boundary_edges"][edge_type]
    boundary1 = part1["boundary_edges"][edge_type]

    assert torch.equal(boundary0["e_id"], torch.tensor([1, 3]))
    assert torch.equal(boundary0["edge_index"], torch.tensor([[1, 3], [2, 0]]))
    assert torch.equal(boundary1["e_id"], torch.tensor([1, 3]))
    assert torch.equal(boundary1["edge_index"], torch.tensor([[1, 3], [2, 0]]))


def test_partition_writer_persists_typed_boundary_edges_in_global_id_space(tmp_path):
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.arange(8, dtype=torch.float32).view(4, 2)},
            "paper": {"x": torch.arange(10, 18, dtype=torch.float32).view(4, 2)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 0], [1, 0, 3, 2, 2]]),
                "weight": torch.tensor([1.0, 2.0, 3.0, 4.0, 9.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 1, 2, 3, 1], [1, 0, 3, 2, 2]]),
                "score": torch.tensor([0.1, 0.2, 0.3, 0.4, 0.9]),
            },
        },
    )

    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    part0 = torch.load(tmp_path / "part-0.pt", weights_only=True)
    part1 = torch.load(tmp_path / "part-1.pt", weights_only=True)

    assert torch.equal(part0["boundary_edges"][writes]["e_id"], torch.tensor([4]))
    assert torch.equal(part0["boundary_edges"][writes]["edge_index"], torch.tensor([[0], [2]]))
    assert torch.equal(part0["boundary_edges"][cites]["e_id"], torch.tensor([4]))
    assert torch.equal(part0["boundary_edges"][cites]["edge_index"], torch.tensor([[1], [2]]))
    assert torch.equal(part1["boundary_edges"][writes]["e_id"], torch.tensor([4]))
    assert torch.equal(part1["boundary_edges"][writes]["edge_index"], torch.tensor([[0], [2]]))
    assert torch.equal(part1["boundary_edges"][cites]["e_id"], torch.tensor([4]))
    assert torch.equal(part1["boundary_edges"][cites]["edge_index"], torch.tensor([[1], [2]]))
