import torch

from vgl.sparse import SparseLayout, from_edge_index
from vgl.sparse.ops import degree, select_rows, spmm


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
