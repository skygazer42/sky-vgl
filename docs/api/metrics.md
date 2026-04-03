# vgl.metrics

评估指标模块。

## Metric 基类

::: vgl.metrics.Metric
    options:
      show_root_heading: true
      show_source: false

## 内置指标

### Accuracy

::: vgl.metrics.Accuracy
    options:
      show_root_heading: true
      show_source: false

### MRR

::: vgl.metrics.MRR
    options:
      show_root_heading: true
      show_source: false

### HitsAtK

::: vgl.metrics.HitsAtK
    options:
      show_root_heading: true
      show_source: false

### FilteredMRR

::: vgl.metrics.FilteredMRR
    options:
      show_root_heading: true
      show_source: false

### FilteredHitsAtK

::: vgl.metrics.FilteredHitsAtK
    options:
      show_root_heading: true
      show_source: false

## 使用方式

指标通过 Task 的 `metrics` 参数指定：

```python
from vgl.tasks import NodeClassificationTask

task = NodeClassificationTask(
    target="y",
    split=("train_mask", "val_mask", "test_mask"),
    metrics=["accuracy"],
)
```

也可以直接实例化指标使用：

```python
from vgl.metrics import Accuracy

metric = Accuracy()
metric.reset()
metric.update(predictions, targets)
result = metric.compute()
```

Metric 的接口包括：

- `reset()` — 重置内部状态
- `update(predictions, targets)` — 更新一批数据
- `compute()` — 计算最终指标值
