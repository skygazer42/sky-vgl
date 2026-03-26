import pytest
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


def test_graph_formats_report_status_and_adjacency_updates_created_formats():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        x=torch.randn(3, 2),
    )

    assert graph.formats() == {"created": ["coo"], "not created": ["csr", "csc"]}

    graph.adjacency(layout="csr")

    assert graph.formats() == {"created": ["coo", "csr"], "not created": ["csc"]}


def test_graph_formats_clone_state_is_isolated_from_base_graph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        x=torch.randn(3, 2),
    )

    clone = graph.formats(["coo", "csr"])

    assert graph.formats() == {"created": ["coo"], "not created": ["csr", "csc"]}
    assert clone.formats() == {"created": ["coo"], "not created": ["csr"]}

    result = clone.create_formats_()

    assert result is None
    assert clone.formats() == {"created": ["coo", "csr"], "not created": []}
    assert graph.formats() == {"created": ["coo"], "not created": ["csr", "csc"]}


def test_graph_formats_can_restrict_to_single_created_format():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        x=torch.randn(3, 2),
    )

    csr_clone = graph.formats("csr")

    assert csr_clone.formats() == {"created": ["csr"], "not created": []}

    csc_csr_clone = graph.formats(["csc", "csr"])

    assert csc_csr_clone.formats() == {"created": ["csr"], "not created": ["csc"]}


def test_graph_formats_validate_requested_formats():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 2),
    )

    with pytest.raises(ValueError):
        graph.formats("bad")

    with pytest.raises(ValueError):
        graph.formats(["coo", "bad"])

    with pytest.raises(ValueError):
        graph.formats([])
