import torch

from vgl import Graph
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


def test_local_graph_shard_maps_local_ids_and_edges_back_to_global_space(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    shard = LocalGraphShard.from_partition_dir(tmp_path, partition_id=1)

    assert torch.equal(shard.local_to_global(torch.tensor([0, 1])), torch.tensor([2, 3]))
    assert torch.equal(shard.global_edge_index(), torch.tensor([[2], [3]]))


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


def test_local_graph_shard_reconstructs_multi_relation_partition_graph(tmp_path):
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
