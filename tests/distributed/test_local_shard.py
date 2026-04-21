import json
from pathlib import Path

import pytest
import torch

from vgl import Graph
import vgl.distributed.shard as distributed_shard_module
import vgl.distributed.store as distributed_store_module
from vgl.distributed.shard import LocalGraphShard
from vgl.distributed.writer import write_partitioned_graph


def test_local_graph_shard_loads_partition_graph_and_store_view(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)
    fetched = shard.feature_store.fetch(("node", "node", "x"), torch.tensor([0, 1])).values

    assert shard.partition.partition_id == 1
    assert torch.equal(shard.node_ids, torch.tensor([2, 3]))
    assert torch.equal(shard.global_to_local(torch.tensor([2, 3])), torch.tensor([0, 1]))
    assert torch.equal(shard.graph.edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(shard.graph.n_id, torch.tensor([2, 3]))
    assert torch.equal(shard.graph_store.edge_index(), torch.tensor([[0], [1]]))
    assert torch.equal(fetched, torch.tensor([[4.0, 5.0], [6.0, 7.0]]))


def test_local_graph_shard_global_to_local_avoids_tensor_tolist(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    def fail_tolist(self):
        raise AssertionError("LocalGraphShard.global_to_local should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    assert torch.equal(shard.global_to_local(torch.tensor([2, 3])), torch.tensor([0, 1]))


def test_local_graph_shard_store_views_reject_mismatched_partition_ids(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)

    with pytest.raises(KeyError, match="partition"):
        shard.feature_store.shape(("node", "node", "x"), partition_id=1)

    with pytest.raises(KeyError, match="partition"):
        shard.feature_store.fetch(("node", "node", "x"), torch.tensor([0]), partition_id=1)

    with pytest.raises(KeyError, match="partition"):
        shard.graph_store.edge_index(partition_id=1)


def test_local_graph_shard_from_partition_dir_rejects_unknown_partition_id(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    with pytest.raises(KeyError, match="unknown partition_id: 9"):
        LocalGraphShard.from_partition_dir(tmp_path, partition_id=9)


def test_local_graph_shard_rejects_parent_escaping_partition_payload_path_before_load(tmp_path, monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["partitions"][0]["path"] = "../escape.pt"
    manifest_path.write_text(json.dumps(payload))

    def fail_load(*args, **kwargs):
        raise AssertionError("torch.load should not be reached for escaped partition payload paths")

    monkeypatch.setattr(distributed_store_module.torch, "load", fail_load)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)

    with pytest.raises(ValueError, match="partition payload path"):
        shard.feature_store.fetch(("node", "node", "x"), torch.tensor([0]))


def test_local_graph_shard_rejects_absolute_partition_payload_path_before_load(tmp_path, monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["partitions"][0]["path"] = str((tmp_path / "part-0.pt").resolve())
    manifest_path.write_text(json.dumps(payload))

    def fail_load(*args, **kwargs):
        raise AssertionError("torch.load should not be reached for absolute partition payload paths")

    monkeypatch.setattr(distributed_store_module.torch, "load", fail_load)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)

    with pytest.raises(ValueError, match="partition payload path"):
        shard.feature_store.fetch(("node", "node", "x"), torch.tensor([0]))


def test_local_graph_shard_normalizes_partition_payload_path_within_root(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    manifest_path = tmp_path / "manifest.json"
    payload = json.loads(manifest_path.read_text())
    payload["partitions"][0]["path"] = str(Path("nested") / ".." / "part-0.pt")
    manifest_path.write_text(json.dumps(payload))

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)
    fetched = shard.feature_store.fetch(("node", "node", "x"), torch.tensor([0, 1])).values

    assert torch.equal(fetched, torch.tensor([[0.0, 1.0], [2.0, 3.0]]))


def test_local_graph_shard_from_partition_dir_is_lazy_until_data_access(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
        edge_data={"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    load_calls = []
    real_torch_load = distributed_shard_module.torch.load

    def counting_load(path, *args, **kwargs):
        load_calls.append(str(path))
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(distributed_shard_module.torch, "load", counting_load)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)

    assert load_calls == []
    assert shard.partition.partition_id == 0
    assert torch.equal(shard.node_ids, torch.tensor([0, 1]))
    assert torch.equal(shard.global_to_local(torch.tensor([0, 1])), torch.tensor([0, 1]))
    assert torch.equal(shard.local_to_global(torch.tensor([0, 1])), torch.tensor([0, 1]))
    assert torch.equal(shard.edge_ids(), torch.tensor([0]))
    assert torch.equal(shard.boundary_edge_ids(), torch.tensor([1, 3]))
    assert torch.equal(shard.incident_edge_ids(), torch.tensor([0, 1, 3]))
    assert torch.equal(shard.local_to_global_edge(torch.tensor([0])), torch.tensor([0]))
    assert torch.equal(shard.global_to_local_edge(torch.tensor([0])), torch.tensor([0]))
    assert load_calls == []

    fetched = shard.feature_store.fetch(("node", "node", "x"), torch.tensor([0, 1])).values

    assert torch.equal(fetched, torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    assert len(load_calls) == 1
    assert load_calls[0].endswith("part-0.pt")

    assert torch.equal(shard.graph.edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(shard.graph.edata["weight"], torch.tensor([1.0]))
    assert torch.equal(shard.graph_store.edge_index(), torch.tensor([[0], [1]]))
    assert torch.equal(shard.boundary_edge_index(), torch.tensor([[1, 3], [2, 0]]))
    assert len(load_calls) == 1


def test_local_graph_shard_maps_local_ids_and_edges_back_to_global_space(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    assert torch.equal(shard.local_to_global(torch.tensor([0, 1])), torch.tensor([2, 3]))
    assert torch.equal(shard.global_edge_index(), torch.tensor([[2], [3]]))


def test_local_graph_shard_global_to_local_rejects_nodes_outside_partition(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    with pytest.raises(KeyError, match="node 1 is not present in partition 1"):
        shard.global_to_local(torch.tensor([1]))


def test_local_graph_shard_global_to_local_rejects_nodes_without_tensor_item(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    def fail_item(self):
        raise AssertionError("LocalGraphShard.global_to_local errors should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    with pytest.raises(KeyError, match="node 1 is not present in partition 1"):
        shard.global_to_local(torch.tensor([1]))


def test_local_graph_shard_global_to_local_rejects_nodes_without_tensor_int(monkeypatch, tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    def fail_int(self):
        raise AssertionError("LocalGraphShard.global_to_local errors should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    with pytest.raises(KeyError, match="node 1 is not present in partition 1"):
        shard.global_to_local(torch.tensor([1]))


def test_local_graph_shard_loads_temporal_partition_graph(tmp_path):
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
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    assert shard.partition.partition_id == 1
    assert shard.graph.schema.time_attr == "timestamp"
    assert torch.equal(shard.node_ids, torch.tensor([2, 3]))
    assert torch.equal(shard.graph.edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(shard.graph.edata["timestamp"], torch.tensor([7]))
    assert torch.equal(shard.local_to_global(torch.tensor([0, 1])), torch.tensor([2, 3]))


def test_local_graph_shard_preserves_temporal_metadata_without_eager_payload_load(monkeypatch, tmp_path):
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
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    load_calls = []
    real_torch_load = distributed_shard_module.torch.load

    def counting_load(path, *args, **kwargs):
        load_calls.append(str(path))
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(distributed_shard_module.torch, "load", counting_load)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    assert load_calls == []
    assert shard.graph.schema.time_attr == "timestamp"
    assert torch.equal(shard.node_ids, torch.tensor([2, 3]))
    assert load_calls == []

    assert torch.equal(shard.graph.edata["timestamp"], torch.tensor([7]))
    assert len(load_calls) == 1
    assert load_calls[0].endswith("part-1.pt")


def test_local_graph_shard_reconstructs_multi_relation_partition_graph_edge_ids(tmp_path):
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
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    assert set(shard.graph.edges) == {follows, likes}
    assert torch.equal(shard.graph.edges[follows].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(shard.graph.edges[likes].edge_index, torch.tensor([[1], [0]]))
    assert torch.equal(shard.graph.edges[follows].weight, torch.tensor([3.0, 4.0]))
    assert torch.equal(shard.graph.edges[likes].score, torch.tensor([0.7]))
    assert torch.equal(shard.global_edge_index(edge_type=follows), torch.tensor([[2, 3], [3, 2]]))
    assert torch.equal(shard.global_edge_index(edge_type=likes), torch.tensor([[3], [2]]))
    assert torch.equal(shard.edge_ids(edge_type=follows), torch.tensor([2, 3]))
    assert torch.equal(shard.global_to_local_edge(torch.tensor([3, 2]), edge_type=follows), torch.tensor([1, 0]))
    assert torch.equal(shard.local_to_global_edge(torch.tensor([0, 1]), edge_type=follows), torch.tensor([2, 3]))


def test_local_graph_shard_global_to_local_edge_avoids_tensor_tolist(monkeypatch, tmp_path):
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
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    def fail_tolist(self):
        raise AssertionError("LocalGraphShard.global_to_local_edge should stay on tensors")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    assert torch.equal(shard.global_to_local_edge(torch.tensor([3, 2]), edge_type=follows), torch.tensor([1, 0]))


def test_local_graph_shard_preserves_manifest_edge_type_order(tmp_path):
    writes = ("author", "writes", "paper")
    written_by = ("paper", "written_by", "author")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.arange(4, dtype=torch.float32).view(2, 2)},
            "paper": {"x": torch.arange(8, 20, dtype=torch.float32).view(6, 2)},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "weight": torch.tensor([1.0, 2.0]),
            },
            written_by: {
                "edge_index": torch.tensor([[1, 2], [0, 1]]),
                "weight": torch.tensor([3.0, 4.0]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 2], [2, 3]]),
                "weight": torch.tensor([5.0, 6.0]),
            },
        },
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=1)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)

    assert tuple(shard.graph.edges) == (writes, written_by, cites)


def test_local_graph_shard_reconstructs_heterogeneous_partition_graph_edge_ids(tmp_path):
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

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    assert set(shard.graph.nodes) == {"author", "paper"}
    assert set(shard.graph.edges) == {writes, cites}
    assert torch.equal(shard.node_ids_for("author"), torch.tensor([2, 3]))
    assert torch.equal(shard.node_ids_for("paper"), torch.tensor([2, 3]))
    assert torch.equal(shard.global_to_local(torch.tensor([2, 3]), node_type="author"), torch.tensor([0, 1]))
    assert torch.equal(shard.local_to_global(torch.tensor([0, 1]), node_type="paper"), torch.tensor([2, 3]))
    assert torch.equal(shard.graph.nodes["paper"].x, torch.tensor([[14.0, 15.0], [16.0, 17.0]]))
    assert torch.equal(shard.graph.edges[writes].edge_index, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(shard.graph.edges[writes].weight, torch.tensor([3.0, 4.0]))
    assert torch.equal(shard.graph.edges[cites].score, torch.tensor([0.3, 0.4]))
    assert torch.equal(shard.global_edge_index(edge_type=writes), torch.tensor([[2, 3], [3, 2]]))
    assert torch.equal(shard.global_edge_index(edge_type=cites), torch.tensor([[2, 3], [3, 2]]))
    assert torch.equal(shard.edge_ids(edge_type=writes), torch.tensor([2, 3]))
    assert torch.equal(shard.global_to_local_edge(torch.tensor([3, 2]), edge_type=writes), torch.tensor([1, 0]))
    assert torch.equal(shard.local_to_global_edge(torch.tensor([0, 1]), edge_type=cites), torch.tensor([2, 3]))


def test_local_graph_shard_node_ids_is_ambiguous_for_multi_type_shards(tmp_path):
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

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    with pytest.raises(AttributeError, match="node_ids is ambiguous for multi-type shards"):
        shard.node_ids


def test_local_graph_shard_requires_edge_type_for_multi_relation_queries(tmp_path):
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

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)

    with pytest.raises(KeyError, match="edge_type is required when multiple edge types exist"):
        shard.edge_ids()

    with pytest.raises(KeyError, match="edge_type is required when multiple edge types exist"):
        shard.boundary_edge_ids()

    with pytest.raises(KeyError, match="edge_type is required when multiple edge types exist"):
        shard.incident_edge_index()


def test_local_graph_shard_exposes_boundary_and_incident_edge_queries(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)

    assert torch.equal(shard.boundary_edge_ids(), torch.tensor([1, 3]))
    assert torch.equal(shard.boundary_edge_index(), torch.tensor([[1, 3], [2, 0]]))
    assert torch.equal(shard.incident_edge_ids(), torch.tensor([0, 1, 3]))
    assert torch.equal(shard.incident_edge_index(), torch.tensor([[0, 1, 3], [1, 2, 0]]))


def test_local_graph_shard_exposes_typed_boundary_and_incident_edge_queries(tmp_path):
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

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=0)

    assert torch.equal(shard.boundary_edge_ids(edge_type=writes), torch.tensor([4]))
    assert torch.equal(shard.boundary_edge_index(edge_type=writes), torch.tensor([[0], [2]]))
    assert torch.equal(shard.incident_edge_ids(edge_type=writes), torch.tensor([0, 1, 4]))
    assert torch.equal(shard.incident_edge_index(edge_type=writes), torch.tensor([[0, 1, 0], [1, 0, 2]]))
    assert torch.equal(shard.boundary_edge_ids(edge_type=cites), torch.tensor([4]))
    assert torch.equal(shard.boundary_edge_index(edge_type=cites), torch.tensor([[1], [2]]))
    assert torch.equal(shard.incident_edge_ids(edge_type=cites), torch.tensor([0, 1, 4]))
    assert torch.equal(shard.incident_edge_index(edge_type=cites), torch.tensor([[0, 1, 1], [1, 0, 2]]))
