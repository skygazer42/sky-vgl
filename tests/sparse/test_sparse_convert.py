import pytest
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


def test_to_coo_from_csr_avoids_repeat_interleave(monkeypatch):
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 0, 1]])

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("CSR to COO conversion should avoid repeat_interleave")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

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


def test_to_csr_preserves_multi_dimensional_values():
    edge_index = torch.tensor([[1, 0, 1], [3, 2, 0]])
    values = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    csr = to_csr(from_edge_index(edge_index, shape=(2, 4), values=values))

    assert csr.layout is SparseLayout.CSR
    assert torch.equal(csr.crow_indices, torch.tensor([0, 1, 3]))
    assert torch.equal(csr.col_indices, torch.tensor([2, 0, 3]))
    assert torch.equal(csr.values, torch.tensor([[2.0, 20.0], [3.0, 30.0], [1.0, 10.0]]))


def test_to_coo_round_trips_multi_dimensional_values_from_csc():
    edge_index = torch.tensor([[1, 0, 1], [3, 2, 0]])
    values = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    csc = to_csc(from_edge_index(edge_index, shape=(2, 4), values=values))
    coo = to_coo(csc)

    assert torch.equal(torch.stack((coo.row, coo.col)), torch.tensor([[1, 0, 1], [0, 2, 3]]))
    assert torch.equal(coo.values, torch.tensor([[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]]))


def test_to_coo_from_csc_avoids_repeat_interleave(monkeypatch):
    edge_index = torch.tensor([[1, 0, 1], [3, 2, 0]])
    values = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("CSC to COO conversion should avoid repeat_interleave")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    csc = to_csc(from_edge_index(edge_index, shape=(2, 4), values=values))
    coo = to_coo(csc)

    assert torch.equal(torch.stack((coo.row, coo.col)), torch.tensor([[1, 0, 1], [0, 2, 3]]))
    assert torch.equal(coo.values, torch.tensor([[3.0, 30.0], [2.0, 20.0], [1.0, 10.0]]))


def test_from_torch_sparse_imports_coo_csr_and_csc_layouts():
    from vgl.sparse import from_torch_sparse

    coo = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [1, 0]]),
        torch.tensor([1.5, 2.5]),
        (3, 3),
    )
    csr = torch.sparse_csr_tensor(
        torch.tensor([0, 1, 2, 2]),
        torch.tensor([1, 0]),
        torch.tensor([1.5, 2.5]),
        size=(3, 3),
    )
    csc = torch.sparse_csc_tensor(
        torch.tensor([0, 1, 2, 2]),
        torch.tensor([2, 0]),
        torch.tensor([2.5, 1.5]),
        size=(3, 3),
    )

    imported_coo = from_torch_sparse(coo)
    imported_csr = from_torch_sparse(csr)
    imported_csc = from_torch_sparse(csc)

    assert imported_coo.layout is SparseLayout.COO
    assert torch.equal(imported_coo.row, torch.tensor([0, 2]))
    assert torch.equal(imported_coo.col, torch.tensor([1, 0]))
    assert torch.equal(imported_coo.values, torch.tensor([1.5, 2.5]))

    assert imported_csr.layout is SparseLayout.CSR
    assert torch.equal(imported_csr.crow_indices, torch.tensor([0, 1, 2, 2]))
    assert torch.equal(imported_csr.col_indices, torch.tensor([1, 0]))
    assert torch.equal(imported_csr.values, torch.tensor([1.5, 2.5]))

    assert imported_csc.layout is SparseLayout.CSC
    assert torch.equal(imported_csc.ccol_indices, torch.tensor([0, 1, 2, 2]))
    assert torch.equal(imported_csc.row_indices, torch.tensor([2, 0]))
    assert torch.equal(imported_csc.values, torch.tensor([2.5, 1.5]))


def test_to_torch_sparse_exports_native_sparse_layouts():
    from vgl.sparse import to_torch_sparse

    coo = from_edge_index(
        torch.tensor([[0, 2], [1, 0]]),
        shape=(3, 3),
        values=torch.tensor([1.5, 2.5]),
    )
    csr = to_csr(coo)
    csc = to_csc(coo)

    exported_coo = to_torch_sparse(coo)
    exported_csr = to_torch_sparse(csr)
    exported_csc = to_torch_sparse(csc)

    assert exported_coo.layout is torch.sparse_coo
    assert torch.equal(exported_coo._indices(), torch.tensor([[0, 2], [1, 0]]))
    assert torch.equal(exported_coo._values(), torch.tensor([1.5, 2.5]))

    assert exported_csr.layout is torch.sparse_csr
    assert torch.equal(exported_csr.crow_indices(), csr.crow_indices)
    assert torch.equal(exported_csr.col_indices(), csr.col_indices)
    assert torch.equal(exported_csr.values(), csr.values)

    assert exported_csc.layout is torch.sparse_csc
    assert torch.equal(exported_csc.ccol_indices(), csc.ccol_indices)
    assert torch.equal(exported_csc.row_indices(), csc.row_indices)
    assert torch.equal(exported_csc.values(), csc.values)


def test_torch_sparse_interop_preserves_multi_dimensional_values():
    from vgl.sparse import from_torch_sparse, to_torch_sparse

    values = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    tensor = torch.sparse_coo_tensor(
        torch.tensor([[0, 2], [1, 0]]),
        values,
        (3, 3, 2),
    )

    sparse = from_torch_sparse(tensor)
    restored = to_torch_sparse(sparse)

    assert sparse.layout is SparseLayout.COO
    assert torch.equal(sparse.values, values)
    assert torch.equal(restored._values(), values)


def test_to_torch_sparse_materializes_unit_values_for_structure_only_sparse():
    from vgl.sparse import to_torch_sparse

    sparse = from_edge_index(torch.tensor([[0, 2], [1, 0]]), shape=(3, 3))

    exported = to_torch_sparse(sparse)

    assert exported.layout is torch.sparse_coo
    assert torch.equal(exported._indices(), torch.tensor([[0, 2], [1, 0]]))
    assert torch.equal(exported._values(), torch.ones(2))


def test_from_torch_sparse_rejects_dense_and_non_matrix_inputs():
    from vgl.sparse import from_torch_sparse

    with pytest.raises(ValueError, match="sparse tensor"):
        from_torch_sparse(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    tensor = torch.sparse_coo_tensor(
        torch.tensor([[0, 1], [1, 0], [0, 0]]),
        torch.tensor([1.0, 2.0]),
        (2, 2, 1),
    )
    with pytest.raises(ValueError, match="two sparse dimensions"):
        from_torch_sparse(tensor)
