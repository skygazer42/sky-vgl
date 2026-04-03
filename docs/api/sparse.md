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
