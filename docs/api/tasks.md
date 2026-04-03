# vgl.tasks

任务定义模块，定义各种图学习任务的监督合约。

## Task 基类

::: vgl.tasks.Task
    options:
      show_root_heading: true
      show_source: false

## 内置任务

### NodeClassificationTask

::: vgl.tasks.NodeClassificationTask
    options:
      show_root_heading: true
      show_source: false

### GraphClassificationTask

::: vgl.tasks.GraphClassificationTask
    options:
      show_root_heading: true
      show_source: false

### LinkPredictionTask

::: vgl.tasks.LinkPredictionTask
    options:
      show_root_heading: true
      show_source: false

### TemporalEventPredictionTask

::: vgl.tasks.TemporalEventPredictionTask
    options:
      show_root_heading: true
      show_source: false

## 鲁棒训练任务

这些任务变体在标准交叉熵之上增加了噪声鲁棒或正则化机制：

### BootstrapTask

::: vgl.tasks.BootstrapTask
    options:
      show_root_heading: true
      show_source: false

### ConfidencePenaltyTask

::: vgl.tasks.ConfidencePenaltyTask
    options:
      show_root_heading: true
      show_source: false

### FloodingTask

::: vgl.tasks.FloodingTask
    options:
      show_root_heading: true
      show_source: false

### GeneralizedCrossEntropyTask

::: vgl.tasks.GeneralizedCrossEntropyTask
    options:
      show_root_heading: true
      show_source: false

### Poly1CrossEntropyTask

::: vgl.tasks.Poly1CrossEntropyTask
    options:
      show_root_heading: true
      show_source: false

### RDropTask

::: vgl.tasks.RDropTask
    options:
      show_root_heading: true
      show_source: false

### SymmetricCrossEntropyTask

::: vgl.tasks.SymmetricCrossEntropyTask
    options:
      show_root_heading: true
      show_source: false
