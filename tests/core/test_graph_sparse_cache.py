import torch

from vgl import Graph
from vgl.sparse import SparseLayout, select_cols, transpose


def test_homo_graph_caches_adjacency_by_layout():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        x=torch.randn(3, 2),
    )

    first = graph.adjacency()
    second = graph.adjacency()
    csr = graph.adjacency(layout="csr")

    assert first is second
    assert first.layout is SparseLayout.COO
    assert csr.layout is SparseLayout.CSR
    assert torch.equal(csr.crow_indices, torch.tensor([0, 1, 3, 3]))


def test_hetero_graph_adjacency_uses_edge_type_shape():
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(2, 2)},
            "paper": {"x": torch.randn(4, 2)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 3]])
            }
        },
    )

    adjacency = graph.adjacency(edge_type=("author", "writes", "paper"))

    assert adjacency.shape == (2, 4)
    assert adjacency.layout is SparseLayout.COO


def test_homo_graph_caches_csc_adjacency_and_supports_sparse_ops():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        x=torch.randn(3, 2),
    )

    first = graph.adjacency(layout="csc")
    second = graph.adjacency(layout=SparseLayout.CSC)
    selected = select_cols(first, torch.tensor([2, 0]))
    transposed = transpose(first)

    assert first is second
    assert first.layout is SparseLayout.CSC
    assert torch.equal(first.ccol_indices, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(first.row_indices, torch.tensor([1, 0, 1]))
    assert selected.shape == (3, 2)
    assert torch.equal(torch.stack((selected.row, selected.col)), torch.tensor([[1, 1], [1, 0]]))
    assert transposed.layout is SparseLayout.CSR
    assert transposed.shape == (3, 3)
    assert torch.equal(transposed.crow_indices, first.ccol_indices)
    assert torch.equal(transposed.col_indices, first.row_indices)
