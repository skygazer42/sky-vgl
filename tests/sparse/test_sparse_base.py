import pytest
import torch


from vgl.sparse.base import SparseLayout, SparseTensor


def test_sparse_layout_exposes_supported_layouts():
    assert SparseLayout.COO.value == "coo"
    assert SparseLayout.CSR.value == "csr"
    assert SparseLayout.CSC.value == "csc"


def test_coo_sparse_tensor_reports_nnz():
    sparse = SparseTensor(
        layout=SparseLayout.COO,
        shape=(3, 4),
        row=torch.tensor([0, 1, 2]),
        col=torch.tensor([1, 2, 3]),
    )

    assert sparse.nnz == 3


def test_coo_sparse_tensor_rejects_out_of_bounds_indices():
    with pytest.raises(ValueError, match="row indices"):
        SparseTensor(
            layout=SparseLayout.COO,
            shape=(2, 2),
            row=torch.tensor([0, 2]),
            col=torch.tensor([1, 0]),
        )


def test_csr_sparse_tensor_rejects_invalid_pointer_shape():
    with pytest.raises(ValueError, match="crow_indices"):
        SparseTensor(
            layout=SparseLayout.CSR,
            shape=(2, 3),
            crow_indices=torch.tensor([0, 1]),
            col_indices=torch.tensor([0]),
        )


def test_csr_and_csc_sparse_tensor_validation_avoid_tensor_item(monkeypatch):
    def fail_item(self):
        raise AssertionError("SparseTensor validation should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    csr = SparseTensor(
        layout=SparseLayout.CSR,
        shape=(2, 3),
        crow_indices=torch.tensor([0, 1, 2]),
        col_indices=torch.tensor([0, 2]),
    )
    csc = SparseTensor(
        layout=SparseLayout.CSC,
        shape=(3, 2),
        ccol_indices=torch.tensor([0, 1, 2]),
        row_indices=torch.tensor([0, 2]),
    )

    assert csr.nnz == 2
    assert csc.nnz == 2


def test_sparse_tensor_accepts_multi_dimensional_values():
    sparse = SparseTensor(
        layout=SparseLayout.COO,
        shape=(3, 4),
        row=torch.tensor([0, 1, 2]),
        col=torch.tensor([1, 2, 3]),
        values=torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
    )

    assert sparse.nnz == 3
    assert sparse.values.shape == (3, 2)


def test_sparse_tensor_rejects_values_with_wrong_leading_dimension():
    with pytest.raises(ValueError, match="values"):
        SparseTensor(
            layout=SparseLayout.COO,
            shape=(3, 4),
            row=torch.tensor([0, 1, 2]),
            col=torch.tensor([1, 2, 3]),
            values=torch.tensor([[1.0, 10.0], [2.0, 20.0]]),
        )
