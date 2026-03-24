import torch

from vgl import Graph
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
    assert torch.equal(part0["node_ids"], torch.tensor([0, 1]))
    assert torch.equal(part1["node_ids"], torch.tensor([2, 3]))
    assert torch.equal(part0["graph"]["edge_index"], torch.tensor([[0], [1]]))
    assert torch.equal(part1["graph"]["edge_index"], torch.tensor([[0], [1]]))
