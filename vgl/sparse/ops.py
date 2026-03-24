import torch

from vgl.sparse.base import SparseLayout, SparseTensor
from vgl.sparse.convert import to_coo


def degree(sparse: SparseTensor, *, dim: int = 0) -> torch.Tensor:
    coo = to_coo(sparse)
    if dim == 0:
        return torch.bincount(coo.row, minlength=coo.shape[0])
    if dim == 1:
        return torch.bincount(coo.col, minlength=coo.shape[1])
    raise ValueError("degree dim must be 0 or 1")


def select_rows(sparse: SparseTensor, rows: torch.Tensor) -> SparseTensor:
    coo = to_coo(sparse)
    rows = rows.to(dtype=torch.long)
    row_map = {int(row): index for index, row in enumerate(rows.tolist())}
    selected_positions = [index for index, row in enumerate(coo.row.tolist()) if int(row) in row_map]
    selected_index = torch.tensor(selected_positions, dtype=torch.long, device=coo.row.device)
    if selected_index.numel() == 0:
        empty = torch.empty(0, dtype=torch.long, device=coo.row.device)
        return SparseTensor(layout=SparseLayout.COO, shape=(rows.numel(), coo.shape[1]), row=empty, col=empty)
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
