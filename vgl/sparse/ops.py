import torch

from vgl.sparse.base import SparseLayout, SparseTensor
from vgl.sparse.convert import to_coo


def _replace_values(sparse: SparseTensor, values: torch.Tensor) -> SparseTensor:
    if sparse.layout is SparseLayout.COO:
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=sparse.shape,
            row=sparse.row,
            col=sparse.col,
            values=values,
        )
    if sparse.layout is SparseLayout.CSR:
        return SparseTensor(
            layout=SparseLayout.CSR,
            shape=sparse.shape,
            crow_indices=sparse.crow_indices,
            col_indices=sparse.col_indices,
            values=values,
        )
    if sparse.layout is SparseLayout.CSC:
        return SparseTensor(
            layout=SparseLayout.CSC,
            shape=sparse.shape,
            ccol_indices=sparse.ccol_indices,
            row_indices=sparse.row_indices,
            values=values,
        )
    raise ValueError(f"Unsupported sparse layout: {sparse.layout}")


def sum(sparse: SparseTensor, *, dim: int = 0) -> torch.Tensor:
    coo = to_coo(sparse)
    if dim == 0:
        index = coo.row
        size = coo.shape[0]
    elif dim == 1:
        index = coo.col
        size = coo.shape[1]
    else:
        raise ValueError("dim must be 0 or 1")

    if coo.values is None:
        weights = torch.ones(coo.nnz, dtype=torch.long, device=index.device)
    else:
        weights = coo.values.reshape(-1)
        if weights.dtype == torch.bool:
            weights = weights.to(dtype=torch.long)

    result = torch.zeros(size, dtype=weights.dtype, device=weights.device)
    if weights.numel() == 0:
        return result
    result.index_add_(0, index, weights)
    return result


def degree(sparse: SparseTensor, *, dim: int = 0) -> torch.Tensor:
    return sum(sparse, dim=dim)


def select_rows(sparse: SparseTensor, rows: torch.Tensor) -> SparseTensor:
    coo = to_coo(sparse)
    rows = rows.to(dtype=torch.long)
    row_map = {int(row): index for index, row in enumerate(rows.tolist())}
    selected_positions = [index for index, row in enumerate(coo.row.tolist()) if int(row) in row_map]
    selected_index = torch.tensor(selected_positions, dtype=torch.long, device=coo.row.device)
    if selected_index.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=coo.row.device)
        empty_values = None
        if coo.values is not None:
            empty_values = coo.values.new_empty((0,))
        return SparseTensor(layout=SparseLayout.COO, shape=(rows.numel(), coo.shape[1]), row=empty, col=empty, values=empty_values)
    selected_row = torch.tensor(
        [row_map[int(row)] for row in coo.row[selected_index].tolist()],
        dtype=torch.long,
        device=coo.row.device,
    )
    selected_col = coo.col[selected_index]
    selected_values = None if coo.values is None else coo.values[selected_index]
    return SparseTensor(
        layout=SparseLayout.COO,
        shape=(rows.numel(), coo.shape[1]),
        row=selected_row,
        col=selected_col,
        values=selected_values,
    )


def select_cols(sparse: SparseTensor, cols: torch.Tensor) -> SparseTensor:
    coo = to_coo(sparse)
    cols = cols.to(dtype=torch.long)
    col_map = {int(col): index for index, col in enumerate(cols.tolist())}
    selected_positions = [index for index, col in enumerate(coo.col.tolist()) if int(col) in col_map]
    selected_index = torch.tensor(selected_positions, dtype=torch.long, device=coo.col.device)
    if selected_index.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=coo.col.device)
        empty_values = None
        if coo.values is not None:
            empty_values = coo.values.new_empty((0,))
        return SparseTensor(layout=SparseLayout.COO, shape=(coo.shape[0], cols.numel()), row=empty, col=empty, values=empty_values)
    selected_row = coo.row[selected_index]
    selected_col = torch.tensor(
        [col_map[int(col)] for col in coo.col[selected_index].tolist()],
        dtype=torch.long,
        device=coo.col.device,
    )
    selected_values = None if coo.values is None else coo.values[selected_index]
    return SparseTensor(
        layout=SparseLayout.COO,
        shape=(coo.shape[0], cols.numel()),
        row=selected_row,
        col=selected_col,
        values=selected_values,
    )


def transpose(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.COO:
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=(sparse.shape[1], sparse.shape[0]),
            row=sparse.col,
            col=sparse.row,
            values=sparse.values,
        )
    if sparse.layout is SparseLayout.CSR:
        return SparseTensor(
            layout=SparseLayout.CSC,
            shape=(sparse.shape[1], sparse.shape[0]),
            ccol_indices=sparse.crow_indices,
            row_indices=sparse.col_indices,
            values=sparse.values,
        )
    if sparse.layout is SparseLayout.CSC:
        return SparseTensor(
            layout=SparseLayout.CSR,
            shape=(sparse.shape[1], sparse.shape[0]),
            crow_indices=sparse.ccol_indices,
            col_indices=sparse.row_indices,
            values=sparse.values,
        )
    raise ValueError(f"Unsupported sparse layout: {sparse.layout}")


def spmm(sparse: SparseTensor, dense: torch.Tensor) -> torch.Tensor:
    coo = to_coo(sparse)
    if dense.ndim != 2:
        raise ValueError("dense input must be rank-2")
    if dense.size(0) != coo.shape[1]:
        raise ValueError("dense input row count must match sparse column dimension")
    result = dense.new_zeros((coo.shape[0], dense.size(1)))
    if coo.nnz == 0:
        return result
    values = coo.values
    source = dense[coo.col]
    if values is not None:
        source = source * values.unsqueeze(-1).to(dtype=source.dtype)
    result.index_add_(0, coo.row, source)
    return result


def sddmm(sparse: SparseTensor, lhs: torch.Tensor, rhs: torch.Tensor) -> SparseTensor:
    coo = to_coo(sparse)
    if lhs.ndim == 0 or rhs.ndim == 0:
        raise ValueError("lhs and rhs must have at least one dimension")
    if lhs.size(0) != coo.shape[0]:
        raise ValueError("lhs row count must match sparse row dimension")
    if rhs.size(0) != coo.shape[1]:
        raise ValueError("rhs row count must match sparse column dimension")
    if lhs.ndim != rhs.ndim:
        raise ValueError("lhs and rhs must have the same rank")
    if lhs.ndim == 1:
        values = lhs[coo.row] * rhs[coo.col]
        return _replace_values(sparse, values)
    if lhs.ndim != 2:
        raise ValueError("lhs and rhs must be rank-1 or rank-2")
    if lhs.size(1) != rhs.size(1):
        raise ValueError("lhs and rhs feature dimensions must match")
    values = (lhs[coo.row] * rhs[coo.col]).sum(dim=-1)
    return _replace_values(sparse, values)


def edge_softmax(sparse: SparseTensor, scores: torch.Tensor, *, dim: int = 1) -> torch.Tensor:
    coo = to_coo(sparse)
    if dim == 0:
        index = coo.row
        size = coo.shape[0]
    elif dim == 1:
        index = coo.col
        size = coo.shape[1]
    else:
        raise ValueError("dim must be 0 or 1")
    if scores.ndim == 0:
        raise ValueError("scores must have at least one dimension")
    if scores.size(0) != coo.nnz:
        raise ValueError("scores length must match sparse nnz")
    if coo.nnz == 0:
        return scores.clone()
    shifted = scores - scores.max()
    weights = shifted.exp()
    normalizer = torch.zeros((size,) + tuple(weights.shape[1:]), dtype=weights.dtype, device=weights.device)
    normalizer.index_add_(0, index, weights)
    return weights / normalizer[index].clamp_min(1e-12)
