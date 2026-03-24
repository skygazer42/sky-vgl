import torch

from vgl.sparse import SparseLayout, select_cols, transpose
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


def test_in_memory_graph_store_builds_csc_adjacency_and_supports_sparse_ops():
    edge_index = torch.tensor([[0, 1, 1], [1, 0, 2]])
    store = InMemoryGraphStore({WRITES: edge_index}, num_nodes={"author": 2, "paper": 3})

    adjacency = store.adjacency(edge_type=WRITES, layout="csc")
    selected = select_cols(adjacency, torch.tensor([2, 0]))
    transposed = transpose(adjacency)

    assert adjacency.layout is SparseLayout.CSC
    assert adjacency.shape == (2, 3)
    assert torch.equal(adjacency.ccol_indices, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(adjacency.row_indices, torch.tensor([1, 0, 1]))
    assert selected.shape == (2, 2)
    assert torch.equal(torch.stack((selected.row, selected.col)), torch.tensor([[1, 1], [1, 0]]))
    assert transposed.layout is SparseLayout.CSR
    assert transposed.shape == (3, 2)
    assert torch.equal(transposed.crow_indices, adjacency.ccol_indices)
    assert torch.equal(transposed.col_indices, adjacency.row_indices)
