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


def _empty_values_like(values: torch.Tensor | None) -> torch.Tensor | None:
    if values is None:
        return None
    return values.new_empty((0,) + tuple(values.shape[1:]))


def _optional_lookup_positions(index_ids: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    index_ids = torch.as_tensor(index_ids, dtype=torch.long).view(-1)
    values = torch.as_tensor(values, dtype=torch.long, device=index_ids.device).view(-1)
    if values.numel() == 0:
        return values
    if index_ids.numel() == 0:
        return torch.full_like(values, -1)

    order = torch.argsort(index_ids, stable=True)
    sorted_index_ids = index_ids[order]
    keep_last = torch.ones(sorted_index_ids.numel(), dtype=torch.bool, device=index_ids.device)
    if sorted_index_ids.numel() > 1:
        keep_last[:-1] = sorted_index_ids[:-1] != sorted_index_ids[1:]
    unique_index_ids = sorted_index_ids[keep_last]
    source_positions = order[keep_last]

    positions = torch.searchsorted(unique_index_ids, values.contiguous())
    matched = positions < unique_index_ids.numel()
    if bool(matched.any()):
        matched_indices = positions[matched]
        valid_matches = torch.zeros_like(matched)
        valid_matches[matched] = unique_index_ids.index_select(0, matched_indices) == values[matched]
        matched = valid_matches
    lookup = torch.full_like(values, -1)
    if bool(matched.any()):
        lookup[matched] = source_positions.index_select(0, positions[matched])
    return lookup


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
        weights = coo.values
        if weights.dtype == torch.bool:
            weights = weights.to(dtype=torch.long)

    result = torch.zeros((size,) + tuple(weights.shape[1:]), dtype=weights.dtype, device=weights.device)
    if coo.nnz == 0:
        return result
    result.index_add_(0, index, weights)
    return result


def degree(sparse: SparseTensor, *, dim: int = 0) -> torch.Tensor:
    return sum(sparse, dim=dim)


def select_rows(sparse: SparseTensor, rows: torch.Tensor) -> SparseTensor:
    coo = to_coo(sparse)
    rows = rows.to(dtype=torch.long)
    selected_row = _optional_lookup_positions(rows, coo.row)
    selected_mask = selected_row >= 0
    if not bool(selected_mask.any()):
        empty = torch.empty(0, dtype=torch.long, device=coo.row.device)
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=(rows.numel(), coo.shape[1]),
            row=empty,
            col=empty,
            values=_empty_values_like(coo.values),
        )
    selected_row = selected_row[selected_mask]
    selected_col = coo.col[selected_mask]
    selected_values = None if coo.values is None else coo.values[selected_mask]
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
    selected_col = _optional_lookup_positions(cols, coo.col)
    selected_mask = selected_col >= 0
    if not bool(selected_mask.any()):
        empty = torch.empty(0, dtype=torch.long, device=coo.col.device)
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=(coo.shape[0], cols.numel()),
            row=empty,
            col=empty,
            values=_empty_values_like(coo.values),
        )
    selected_row = coo.row[selected_mask]
    selected_col = selected_col[selected_mask]
    selected_values = None if coo.values is None else coo.values[selected_mask]
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

    values = coo.values
    if values is None or values.ndim == 1:
        result = dense.new_zeros((coo.shape[0], dense.size(1)))
    else:
        result = dense.new_zeros((coo.shape[0],) + tuple(values.shape[1:]) + (dense.size(1),))
    if coo.nnz == 0:
        return result

    source = dense[coo.col]
    if values is not None:
        if values.ndim == 1:
            source = source * values.unsqueeze(-1).to(dtype=source.dtype)
        else:
            payload = values.to(dtype=source.dtype).unsqueeze(-1)
            source = source.reshape((source.size(0),) + (1,) * (values.ndim - 1) + (source.size(1),))
            source = payload * source
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
    if lhs.shape[1:-1] != rhs.shape[1:-1]:
        raise ValueError("lhs and rhs payload dimensions must match")
    if lhs.size(-1) != rhs.size(-1):
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
    shifted = scores - scores.amax(dim=0)
    weights = shifted.exp()
    normalizer = torch.zeros((size,) + tuple(weights.shape[1:]), dtype=weights.dtype, device=weights.device)
    normalizer.index_add_(0, index, weights)
    return weights / normalizer[index].clamp_min(1e-12)
