import torch

from vgl.sparse import SparseLayout, from_edge_index, to_coo, to_csc, to_csr


def test_from_edge_index_builds_coo_sparse_tensor():
    edge_index = torch.tensor([[0, 1, 1], [2, 0, 3]])

    sparse = from_edge_index(edge_index, shape=(2, 4))

    assert sparse.layout is SparseLayout.COO
    assert sparse.nnz == 3
    assert torch.equal(sparse.row, edge_index[0])
    assert torch.equal(sparse.col, edge_index[1])


def test_to_csr_converts_coo_tensor():
    edge_index = torch.tensor([[1, 0, 1], [3, 2, 0]])

    csr = to_csr(from_edge_index(edge_index, shape=(2, 4)))

    assert csr.layout is SparseLayout.CSR
    assert torch.equal(csr.crow_indices, torch.tensor([0, 1, 3]))
    assert torch.equal(csr.col_indices, torch.tensor([2, 0, 3]))


def test_to_coo_round_trips_from_csr():
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 1]])

    csr = from_edge_index(edge_index, shape=(3, 3), layout=SparseLayout.CSR)
    coo = to_coo(csr)

    assert torch.equal(torch.stack((coo.row, coo.col)), edge_index)


def test_to_csc_converts_coo_tensor_with_values():
    edge_index = torch.tensor([[1, 0, 1], [3, 2, 0]])
    values = torch.tensor([1.5, 2.0, 3.5])

    csc = to_csc(from_edge_index(edge_index, shape=(2, 4), values=values))

    assert csc.layout is SparseLayout.CSC
    assert torch.equal(csc.ccol_indices, torch.tensor([0, 1, 1, 2, 3]))
    assert torch.equal(csc.row_indices, torch.tensor([1, 0, 1]))
    assert torch.equal(csc.values, torch.tensor([3.5, 2.0, 1.5]))
