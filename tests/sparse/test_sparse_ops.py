import pytest
import torch

from vgl.sparse import SparseLayout, from_edge_index
from vgl.sparse.ops import degree, select_cols, select_rows, spmm, sum as sparse_sum, transpose


def test_degree_counts_edges_by_dimension():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
    )

    assert torch.equal(degree(sparse, dim=0), torch.tensor([2, 0, 1]))
    assert torch.equal(degree(sparse, dim=1), torch.tensor([1, 1, 1]))


def test_select_rows_returns_reindexed_sparse_tensor():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
    )

    selected = select_rows(sparse, torch.tensor([0, 2]))

    assert selected.layout is SparseLayout.COO
    assert selected.shape == (2, 3)
    assert torch.equal(torch.stack((selected.row, selected.col)), torch.tensor([[0, 0, 1], [1, 2, 0]]))


def test_spmm_multiplies_sparse_by_dense_features():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
    )
    dense = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])

    result = spmm(sparse, dense)

    assert torch.equal(result, torch.tensor([[1.0, 2.0], [0.0, 0.0], [1.0, 0.0]]))


def test_select_cols_returns_reindexed_sparse_tensor_with_values():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
        values=torch.tensor([1.0, 2.0, 3.0]),
    )

    selected = select_cols(sparse, torch.tensor([2, 0]))

    assert selected.layout is SparseLayout.COO
    assert selected.shape == (3, 2)
    assert torch.equal(torch.stack((selected.row, selected.col)), torch.tensor([[0, 2], [0, 1]]))
    assert torch.equal(selected.values, torch.tensor([2.0, 3.0]))


def test_select_cols_returns_empty_sparse_tensor_for_empty_selection():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
    )

    selected = select_cols(sparse, torch.tensor([], dtype=torch.long))

    assert selected.layout is SparseLayout.COO
    assert selected.shape == (3, 0)
    assert selected.nnz == 0


def test_transpose_swaps_coo_indices_and_shape():
    sparse = from_edge_index(
        torch.tensor([[0, 2], [1, 0]]),
        shape=(3, 2),
        values=torch.tensor([1.0, 4.0]),
    )

    transposed = transpose(sparse)

    assert transposed.layout is SparseLayout.COO
    assert transposed.shape == (2, 3)
    assert torch.equal(torch.stack((transposed.row, transposed.col)), torch.tensor([[1, 0], [0, 2]]))
    assert torch.equal(transposed.values, torch.tensor([1.0, 4.0]))


def test_transpose_swaps_compressed_layouts():
    edge_index = torch.tensor([[1, 0, 1], [3, 2, 0]])
    values = torch.tensor([1.5, 2.0, 3.5])

    csr = from_edge_index(edge_index, shape=(2, 4), layout=SparseLayout.CSR, values=values)
    csc = from_edge_index(edge_index, shape=(2, 4), layout=SparseLayout.CSC, values=values)

    transposed_csr = transpose(csr)
    transposed_csc = transpose(csc)

    assert transposed_csr.layout is SparseLayout.CSC
    assert transposed_csr.shape == (4, 2)
    assert torch.equal(transposed_csr.ccol_indices, torch.tensor([0, 1, 3]))
    assert torch.equal(transposed_csr.row_indices, torch.tensor([2, 0, 3]))
    assert torch.equal(transposed_csr.values, torch.tensor([2.0, 3.5, 1.5]))

    assert transposed_csc.layout is SparseLayout.CSR
    assert transposed_csc.shape == (4, 2)
    assert torch.equal(transposed_csc.crow_indices, torch.tensor([0, 1, 1, 2, 3]))
    assert torch.equal(transposed_csc.col_indices, torch.tensor([1, 0, 1]))
    assert torch.equal(transposed_csc.values, torch.tensor([3.5, 2.0, 1.5]))


def test_sum_reduces_sparse_values_by_dimension():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
        values=torch.tensor([0.25, 0.75, 2.0]),
    )

    assert torch.equal(sparse_sum(sparse, dim=0), torch.tensor([1.0, 0.0, 2.0]))
    assert torch.equal(sparse_sum(sparse, dim=1), torch.tensor([2.0, 0.25, 0.75]))


def test_sum_counts_structure_when_sparse_tensor_has_no_values():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
    )

    assert torch.equal(sparse_sum(sparse, dim=0), torch.tensor([2, 0, 1]))
    assert torch.equal(sparse_sum(sparse, dim=1), torch.tensor([1, 1, 1]))


def test_sum_rejects_invalid_dimension():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
    )

    with pytest.raises(ValueError, match="dim must be 0 or 1"):
        sparse_sum(sparse, dim=2)


def test_degree_uses_values_when_present():
    sparse = from_edge_index(
        torch.tensor([[0, 0, 2], [1, 2, 0]]),
        shape=(3, 3),
        values=torch.tensor([0.25, 0.75, 2.0]),
    )

    assert torch.equal(degree(sparse, dim=0), torch.tensor([1.0, 0.0, 2.0]))
    assert torch.equal(degree(sparse, dim=1), torch.tensor([2.0, 0.25, 0.75]))
