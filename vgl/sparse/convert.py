import torch

from vgl.sparse.base import SparseLayout, SparseTensor


def _expand_interval_values(values: torch.Tensor, counts: torch.Tensor, *, step: int) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    counts = torch.as_tensor(counts, dtype=torch.long, device=values.device).view(-1)
    if values.numel() == 0 or counts.numel() == 0:
        return values.new_empty(0)

    positive = counts > 0
    values = values[positive]
    counts = counts[positive]
    if values.numel() == 0:
        return values.new_empty(0)

    offsets = torch.cumsum(counts, dim=0) - counts
    base = values - step * offsets
    deltas = torch.empty_like(base)
    deltas[0] = base[0]
    if base.numel() > 1:
        deltas[1:] = base[1:] - base[:-1]
    markers = torch.zeros(counts.sum(), dtype=values.dtype, device=values.device)
    markers[offsets] = deltas
    expanded = torch.cumsum(markers, dim=0)
    if step != 0:
        expanded = expanded + step * torch.arange(counts.sum(), dtype=values.dtype, device=values.device)
    return expanded


def _sparse_shape_from_torch(tensor: torch.Tensor) -> tuple[int, int]:
    if not hasattr(tensor, "sparse_dim") or int(tensor.sparse_dim()) != 2:
        raise ValueError("torch sparse tensor must have exactly two sparse dimensions")
    return int(tensor.shape[0]), int(tensor.shape[1])


def _coo_indices_and_values(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Preserve duplicate entries and current ordering for uncoalesced COO inputs.
    if bool(tensor.is_coalesced()):
        return tensor.indices(), tensor.values()
    return tensor._indices(), tensor._values()


def _explicit_values_for_export(sparse: SparseTensor) -> torch.Tensor:
    if sparse.values is not None:
        return sparse.values
    if sparse.layout is SparseLayout.COO:
        row = sparse.row
        if row is None:
            raise ValueError("row is required for coo layout")
        device = row.device
    elif sparse.layout is SparseLayout.CSR:
        crow_indices = sparse.crow_indices
        if crow_indices is None:
            raise ValueError("crow_indices is required for csr layout")
        device = crow_indices.device
    else:
        ccol_indices = sparse.ccol_indices
        if ccol_indices is None:
            raise ValueError("ccol_indices is required for csc layout")
        device = ccol_indices.device
    return torch.ones(sparse.nnz, dtype=torch.float32, device=device)


def from_torch_sparse(tensor: torch.Tensor) -> SparseTensor:
    layout = tensor.layout
    if layout is torch.sparse_coo:
        shape = _sparse_shape_from_torch(tensor)
        indices, values = _coo_indices_and_values(tensor)
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=shape,
            row=indices[0],
            col=indices[1],
            values=values,
        )
    if layout is torch.sparse_csr:
        shape = _sparse_shape_from_torch(tensor)
        return SparseTensor(
            layout=SparseLayout.CSR,
            shape=shape,
            crow_indices=tensor.crow_indices(),
            col_indices=tensor.col_indices(),
            values=tensor.values(),
        )
    if layout is torch.sparse_csc:
        shape = _sparse_shape_from_torch(tensor)
        return SparseTensor(
            layout=SparseLayout.CSC,
            shape=shape,
            ccol_indices=tensor.ccol_indices(),
            row_indices=tensor.row_indices(),
            values=tensor.values(),
        )
    raise ValueError("from_torch_sparse expects a torch sparse tensor in COO, CSR, or CSC layout")


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
        return to_csc(coo)
    raise ValueError(f"Unsupported sparse layout: {layout}")


def to_coo(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.COO:
        return sparse
    if sparse.layout is SparseLayout.CSR:
        crow_indices = sparse.crow_indices
        col_indices = sparse.col_indices
        if crow_indices is None or col_indices is None:
            raise ValueError("crow_indices and col_indices are required for csr layout")
        counts = crow_indices[1:] - crow_indices[:-1]
        row = _expand_interval_values(
            torch.arange(sparse.shape[0], dtype=torch.long, device=crow_indices.device),
            counts,
            step=0,
        )
        return SparseTensor(
            layout=SparseLayout.COO,
            shape=sparse.shape,
            row=row,
            col=col_indices,
            values=sparse.values,
        )
    ccol_indices = sparse.ccol_indices
    row_indices = sparse.row_indices
    if ccol_indices is None or row_indices is None:
        raise ValueError("ccol_indices and row_indices are required for csc layout")
    counts = ccol_indices[1:] - ccol_indices[:-1]
    col = _expand_interval_values(
        torch.arange(sparse.shape[1], dtype=torch.long, device=ccol_indices.device),
        counts,
        step=0,
    )
    return SparseTensor(
        layout=SparseLayout.COO,
        shape=sparse.shape,
        row=row_indices,
        col=col,
        values=sparse.values,
    )


def to_csr(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.CSR:
        return sparse
    coo = to_coo(sparse)
    row = coo.row
    col = coo.col
    if row is None or col is None:
        raise ValueError("row and col are required for coo layout")
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


def to_csc(sparse: SparseTensor) -> SparseTensor:
    if sparse.layout is SparseLayout.CSC:
        return sparse
    coo = to_coo(sparse)
    row = coo.row
    col = coo.col
    if row is None or col is None:
        raise ValueError("row and col are required for coo layout")
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


def to_torch_sparse(sparse: SparseTensor) -> torch.Tensor:
    values = _explicit_values_for_export(sparse)
    size = tuple(sparse.shape) + tuple(values.shape[1:])
    if sparse.layout is SparseLayout.COO:
        row = sparse.row
        col = sparse.col
        if row is None or col is None:
            raise ValueError("row and col are required for coo layout")
        indices = torch.stack((row, col))
        return torch.sparse_coo_tensor(indices, values, size=size)
    if sparse.layout is SparseLayout.CSR:
        crow_indices = sparse.crow_indices
        col_indices = sparse.col_indices
        if crow_indices is None or col_indices is None:
            raise ValueError("crow_indices and col_indices are required for csr layout")
        return torch.sparse_csr_tensor(
            crow_indices,
            col_indices,
            values,
            size=size,
        )
    if sparse.layout is SparseLayout.CSC:
        ccol_indices = sparse.ccol_indices
        row_indices = sparse.row_indices
        if ccol_indices is None or row_indices is None:
            raise ValueError("ccol_indices and row_indices are required for csc layout")
        return torch.sparse_csc_tensor(
            ccol_indices,
            row_indices,
            values,
            size=size,
        )
    raise ValueError(f"Unsupported sparse layout: {sparse.layout}")
