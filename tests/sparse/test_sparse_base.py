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
