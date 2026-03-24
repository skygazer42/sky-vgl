import torch

from vgl.sparse.base import SparseLayout, SparseTensor


def from_edge_index(
    edge_index: torch.Tensor,
    *,
    shape: tuple[int, int],
    layout: SparseLayout = SparseLayout.COO,
    values: torch.Tensor | None = None,
) -> SparseTensor:
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    row = edge_index[0].to(dtype=torch.long)
    col = edge_index[1].to(dtype=torch.long)
    coo = SparseTensor(layout=SparseLayout.COO, shape=shape, row=row, col=col, values=values)
    if layout is SparseLayout.COO:
        return coo
    if layout is SparseLayout.CSR:
        return to_csr(coo)
    if layout is SparseLayout.CSC:
        return _to_csc(coo)
    raise ValueError(f"Unsupported sparse layout: {layout}")


def to_coo(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.COO:
        return sparse
    if sparse.layout is SparseLayout.CSR:
        counts = sparse.crow_indices[1:] - sparse.crow_indices[:-1]
        row = torch.repeat_interleave(
            torch.arange(sparse.shape[0], dtype=torch.long, device=sparse.crow_indices.device),
            counts,
        )
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=sparse.shape,
            row=row,
            col=sparse.col_indices,
            values=sparse.values,
        )
    counts = sparse.ccol_indices[1:] - sparse.ccol_indices[:-1]
    col = torch.repeat_interleave(
        torch.arange(sparse.shape[1], dtype=torch.long, device=sparse.ccol_indices.device),
        counts,
    )
    return SparseTensor(
        layout=SparseLayout.COO,
        shape=sparse.shape,
        row=sparse.row_indices,
        col=col,
        values=sparse.values,
    )


def to_csr(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.CSR:
        return sparse
    coo = to_coo(sparse)
    row = coo.row
    col = coo.col
    sort_key = row * max(coo.shape[1], 1) + col
    order = torch.argsort(sort_key)
    row = row[order]
    col = col[order]
    values = None if coo.values is None else coo.values[order]
    counts = torch.bincount(row, minlength=coo.shape[0])
    crow_indices = torch.zeros(coo.shape[0] + 1, dtype=torch.long, device=row.device)
    crow_indices[1:] = torch.cumsum(counts, dim=0)
    return SparseTensor(
        layout=SparseLayout.CSR,
        shape=coo.shape,
        crow_indices=crow_indices,
        col_indices=col,
        values=values,
    )


def _to_csc(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.CSC:
        return sparse
    coo = to_coo(sparse)
    row = coo.row
    col = coo.col
    sort_key = col * max(coo.shape[0], 1) + row
    order = torch.argsort(sort_key)
    row = row[order]
    col = col[order]
    values = None if coo.values is None else coo.values[order]
    counts = torch.bincount(col, minlength=coo.shape[1])
    ccol_indices = torch.zeros(coo.shape[1] + 1, dtype=torch.long, device=col.device)
    ccol_indices[1:] = torch.cumsum(counts, dim=0)
    return SparseTensor(
        layout=SparseLayout.CSC,
        shape=coo.shape,
        ccol_indices=ccol_indices,
        row_indices=row,
        values=values,
    )
