# vgl.sparse

稀疏张量模块，提供 COO/CSR/CSC 稀疏布局和高效稀疏运算。

## SparseTensor

`SparseTensor` 是 VGL 的底层稀疏执行引擎，支持：

- COO、CSR、CSC 三种布局及相互转换
- 与 `torch.sparse_*` 原生张量的双向互操作
- 稀疏边负载（payload），形状 `(nnz, ...)`
- 转置、加法归约、行/列结构选择

::: vgl.sparse.SparseTensor
    options:
      show_root_heading: true
      show_source: false

## 稀疏运算

### spmm

稀疏-稠密矩阵乘，保留稀疏 payload 的尾部维度：

```python
from vgl.sparse import spmm

out = spmm(sparse_adj, dense_features)
```

::: vgl.sparse.spmm
    options:
      show_root_heading: true
      show_source: false

### sddmm

采样稠密-稠密矩阵乘：

```python
from vgl.sparse import sddmm

out = sddmm(sparse_adj, query, key)
```

::: vgl.sparse.sddmm
    options:
      show_root_heading: true
      show_source: false

### edge_softmax

边级别 softmax 归一化：

```python
from vgl.sparse import edge_softmax

normalized = edge_softmax(sparse_adj, edge_scores)
```

::: vgl.sparse.edge_softmax
    options:
      show_root_heading: true
      show_source: false

## 布局与转换

### SparseLayout

标识 COO / CSR / CSC 三种内部布局,可作为 `to_coo/to_csr/to_csc` 的目标,也用于 `from_torch_sparse` 自动识别源布局。

::: vgl.sparse.SparseLayout
    options:
      show_root_heading: true
      show_source: false

### from_edge_index

从 `edge_index`(形状 `(2, E)`)直接构造 `SparseTensor`,可携带 payload。

::: vgl.sparse.from_edge_index
    options:
      show_root_heading: true
      show_source: false

### to_coo / to_csr / to_csc

就地或复制地切换布局。布局切换是免拷贝的(懒构建索引),新布局缓存在原 `SparseTensor` 上。

::: vgl.sparse.to_coo
    options:
      show_root_heading: true
      show_source: false

::: vgl.sparse.to_csr
    options:
      show_root_heading: true
      show_source: false

::: vgl.sparse.to_csc
    options:
      show_root_heading: true
      show_source: false

### torch.sparse 互操作

```python
import torch
from vgl.sparse import from_torch_sparse, to_torch_sparse

native = torch.sparse_csr_tensor(crow, col, values, size=(N, N))
vst = from_torch_sparse(native)           # → vgl.sparse.SparseTensor
back = to_torch_sparse(vst)               # → torch.Tensor
```

::: vgl.sparse.from_torch_sparse
    options:
      show_root_heading: true
      show_source: false

::: vgl.sparse.to_torch_sparse
    options:
      show_root_heading: true
      show_source: false

## 其他运算

### degree / sum / transpose

::: vgl.sparse.degree
    options:
      show_root_heading: true
      show_source: false

::: vgl.sparse.sum
    options:
      show_root_heading: true
      show_source: false

::: vgl.sparse.transpose
    options:
      show_root_heading: true
      show_source: false

### select_rows / select_cols

按行或列索引裁剪稀疏结构,同步保留 payload。

::: vgl.sparse.select_rows
    options:
      show_root_heading: true
      show_source: false

::: vgl.sparse.select_cols
    options:
      show_root_heading: true
      show_source: false
