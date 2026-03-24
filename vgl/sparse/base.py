from dataclasses import dataclass
from enum import Enum

import torch


class SparseLayout(str, Enum):
    COO = "coo"
    CSR = "csr"
    CSC = "csc"


@dataclass(frozen=True, slots=True)
class SparseTensor:
    layout: SparseLayout
    shape: tuple[int, int]
    row: torch.Tensor | None = None
    col: torch.Tensor | None = None
    crow_indices: torch.Tensor | None = None
    col_indices: torch.Tensor | None = None
    ccol_indices: torch.Tensor | None = None
    row_indices: torch.Tensor | None = None
    values: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if len(self.shape) != 2:
            raise ValueError("SparseTensor shape must be rank-2")
        if self.shape[0] < 0 or self.shape[1] < 0:
            raise ValueError("SparseTensor shape must be non-negative")
        if self.layout is SparseLayout.COO:
            self._validate_coo()
        elif self.layout is SparseLayout.CSR:
            self._validate_csr()
        elif self.layout is SparseLayout.CSC:
            self._validate_csc()
        else:
            raise ValueError(f"Unsupported sparse layout: {self.layout}")

    @property
    def nnz(self) -> int:
        if self.layout is SparseLayout.COO:
            return int(self.row.numel())
        if self.layout is SparseLayout.CSR:
            return int(self.col_indices.numel())
        return int(self.row_indices.numel())

    def _validate_vector(self, value: torch.Tensor | None, *, name: str) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} is required for {self.layout.value} layout")
        if value.ndim != 1:
            raise ValueError(f"{name} must be rank-1")
        return value.to(dtype=torch.long)

    def _validate_values(self, nnz: int) -> None:
        if self.values is not None and self.values.numel() != nnz:
            raise ValueError("values must match sparse nnz")

    def _validate_coo(self) -> None:
        row = self._validate_vector(self.row, name="row")
        col = self._validate_vector(self.col, name="col")
        if row.numel() != col.numel():
            raise ValueError("row and col must have the same length")
        if row.numel() > 0 and ((row < 0).any() or (row >= self.shape[0]).any()):
            raise ValueError("row indices must fall within shape bounds")
        if col.numel() > 0 and ((col < 0).any() or (col >= self.shape[1]).any()):
            raise ValueError("col indices must fall within shape bounds")
        self._validate_values(int(row.numel()))

    def _validate_csr(self) -> None:
        crow_indices = self._validate_vector(self.crow_indices, name="crow_indices")
        col_indices = self._validate_vector(self.col_indices, name="col_indices")
        if crow_indices.numel() != self.shape[0] + 1:
            raise ValueError("crow_indices must have length rows + 1")
        if crow_indices[0].item() != 0:
            raise ValueError("crow_indices must start at 0")
        if not bool((crow_indices[1:] >= crow_indices[:-1]).all()):
            raise ValueError("crow_indices must be non-decreasing")
        if crow_indices[-1].item() != col_indices.numel():
            raise ValueError("crow_indices must terminate at nnz")
        if col_indices.numel() > 0 and ((col_indices < 0).any() or (col_indices >= self.shape[1]).any()):
            raise ValueError("col_indices must fall within shape bounds")
        self._validate_values(int(col_indices.numel()))

    def _validate_csc(self) -> None:
        ccol_indices = self._validate_vector(self.ccol_indices, name="ccol_indices")
        row_indices = self._validate_vector(self.row_indices, name="row_indices")
        if ccol_indices.numel() != self.shape[1] + 1:
            raise ValueError("ccol_indices must have length cols + 1")
        if ccol_indices[0].item() != 0:
            raise ValueError("ccol_indices must start at 0")
        if not bool((ccol_indices[1:] >= ccol_indices[:-1]).all()):
            raise ValueError("ccol_indices must be non-decreasing")
        if ccol_indices[-1].item() != row_indices.numel():
            raise ValueError("ccol_indices must terminate at nnz")
        if row_indices.numel() > 0 and ((row_indices < 0).any() or (row_indices >= self.shape[0]).any()):
            raise ValueError("row_indices must fall within shape bounds")
        self._validate_values(int(row_indices.numel()))
