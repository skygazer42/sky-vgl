import torch

from vgl import Graph
from vgl.distributed.coordinator import LocalSamplingCoordinator
from vgl.distributed.shard import LocalGraphShard
from vgl.distributed.writer import write_partitioned_graph


NODE_KEY = ("node", "node", "x")


def test_local_sampling_coordinator_routes_seeds_and_fetches_features(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)
    node_ids = torch.tensor([3, 0, 2])

    routes = coordinator.route_node_ids(node_ids)
    fetched = coordinator.fetch_node_features(NODE_KEY, node_ids)

    assert len(routes) == 2
    assert routes[0].partition_id == 0
    assert torch.equal(routes[0].global_ids, torch.tensor([0]))
    assert torch.equal(routes[0].local_ids, torch.tensor([0]))
    assert routes[1].partition_id == 1
    assert torch.equal(routes[1].global_ids, torch.tensor([3, 2]))
    assert torch.equal(routes[1].local_ids, torch.tensor([1, 0]))
    assert torch.equal(fetched.index, node_ids)
    assert torch.equal(fetched.values, torch.tensor([[6.0, 7.0], [0.0, 1.0], [4.0, 5.0]]))


def test_local_sampling_coordinator_exposes_partition_graph_queries(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)

    partition_node_ids = coordinator.partition_node_ids(1)
    local_edge_index = coordinator.fetch_partition_edge_index(1)
    global_edge_index = coordinator.fetch_partition_edge_index(1, global_ids=True)
    adjacency = coordinator.fetch_partition_adjacency(1, layout="csr")

    assert torch.equal(partition_node_ids, torch.tensor([2, 3]))
    assert torch.equal(local_edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(global_edge_index, torch.tensor([[2], [3]]))
    assert adjacency.layout.value == "csr"
    assert adjacency.shape == (2, 2)
    assert torch.equal(adjacency.crow_indices, torch.tensor([0, 1, 1]))
    assert torch.equal(adjacency.col_indices, torch.tensor([1]))


def test_local_sampling_coordinator_exposes_multi_relation_partition_queries(tmp_path):
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
    shards = {
        0: LocalGraphShard.from_partition_dir(tmp_path, partition_id=0),
        1: LocalGraphShard.from_partition_dir(tmp_path, partition_id=1),
    }
    coordinator = LocalSamplingCoordinator(shards)

    local_follows = coordinator.fetch_partition_edge_index(1, edge_type=follows)
    global_likes = coordinator.fetch_partition_edge_index(1, edge_type=likes, global_ids=True)
    adjacency = coordinator.fetch_partition_adjacency(1, edge_type=follows, layout="csc")

    assert torch.equal(local_follows, torch.tensor([[0, 1], [1, 0]]))
    assert torch.equal(global_likes, torch.tensor([[3], [2]]))
    assert adjacency.layout.value == "csc"
    assert adjacency.shape == (2, 2)
