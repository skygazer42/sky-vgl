import torch

from vgl.sparse import SparseLayout
from vgl.storage import InMemoryGraphStore


HOMO_EDGE = ("node", "to", "node")
WRITES = ("author", "writes", "paper")


def test_in_memory_graph_store_returns_edge_indices_and_counts():
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    store = InMemoryGraphStore({HOMO_EDGE: edge_index}, num_nodes={"node": 3})

    assert store.edge_types == (HOMO_EDGE,)
    assert torch.equal(store.edge_index(), edge_index)
    assert store.edge_count() == 3


def test_in_memory_graph_store_builds_type_aware_adjacency():
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    store = InMemoryGraphStore({WRITES: edge_index}, num_nodes={"author": 2, "paper": 3})

    adjacency = store.adjacency(edge_type=WRITES, layout="coo")

    assert adjacency.layout is SparseLayout.COO
    assert adjacency.shape == (2, 3)
    assert adjacency.row.tolist() == [0, 1, 1]
    assert adjacency.col.tolist() == [1, 0, 2]
