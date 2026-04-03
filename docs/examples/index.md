---
hide:
  - toc
---

# 示例

按任务类型索引的完整代码示例。

## 按任务类型

<div class="grid" markdown>

<div class="card" markdown>

### :material-tag: 节点分类

- Cora 全图 GCN
- Citeseer 全图 GAT
- 邻居采样大图训练

[:octicons-arrow-right-24: 节点分类教程](../guide/node-classification.md)

</div>

<div class="card" markdown>

### :material-chart-bubble: 图分类

- MUTAG GIN 分类
- PROTEINS 子图采样

[:octicons-arrow-right-24: 图分类教程](../guide/graph-classification.md)

</div>

<div class="card" markdown>

### :material-link-variant: 链接预测

- 同构图链接预测
- 负采样策略

[:octicons-arrow-right-24: 链接预测教程](../guide/link-prediction.md)

</div>

<div class="card" markdown>

### :material-clock-outline: 时序预测

- 时序事件预测
- 时序邻居采样

[:octicons-arrow-right-24: 时序图教程](../guide/temporal.md)

</div>

</div>

## 卷积层速查

VGL 内置 60+ 种图卷积层，查看 [卷积层速查表](conv-zoo.md) 快速找到适合的算子。

## 示例代码目录

项目仓库中的 `examples/` 目录按图类型组织：

```
examples/
├── homo/        # 同构图示例
├── hetero/      # 异构图示例
└── temporal/    # 时序图示例
```
