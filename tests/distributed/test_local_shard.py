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
