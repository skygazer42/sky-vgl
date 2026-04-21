# vgl.engine

训练引擎模块，包含 Trainer、Callback、Logger 和 Checkpoint 工具。

## Trainer

::: vgl.engine.Trainer
    options:
      show_root_heading: true
      show_source: false

## TrainingHistory

::: vgl.engine.TrainingHistory
    options:
      show_root_heading: true
      show_source: false

## Callback

::: vgl.engine.Callback
    options:
      show_root_heading: true
      show_source: false

### 内置 Callback

::: vgl.engine.EarlyStopping
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.HistoryLogger
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.ModelCheckpoint
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.StopTraining
    options:
      show_root_heading: true
      show_source: false

## Logger

::: vgl.engine.Logger
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.ConsoleLogger
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.JSONLinesLogger
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.CSVLogger
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.TensorBoardLogger
    options:
      show_root_heading: true
      show_source: false

## Checkpoint 工具

```python
from vgl.engine import load_checkpoint, restore_checkpoint, save_checkpoint
from vgl.engine import CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_VERSION
```

::: vgl.engine.load_checkpoint
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.restore_checkpoint
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.save_checkpoint
    options:
      show_root_heading: true
      show_source: false

### 恢复契约

`restore_checkpoint` 在把磁盘 payload 映射回 `Trainer` / `TrainingHistory` 时会执行一轮严格的规范化与一致性校验:

- `model_state_dict` 必须是映射类型,缺失或错型直接报错,不隐式跳过。
- `trainer_state` 与 `history` 的 `completed_epochs`、`global_step`、`best_epoch`、`best_metric`、`active_monitor`、`stop_reason` 之间的关系会逐字段交叉校验;不自洽的 payload 拒绝加载,避免恢复出"半完成"训练。
- 若历史中包含 `final_train`/`final_val` 摘要,它们必须与最后一段 summary 完全一致;被监控(`val_monitor`)的运行还要求 val 历史完整。
- 返回的 `resume_state` 与 checkpoint payload 彼此独立(已 detach),修改一方不会污染另一方。

在 `Trainer` 级别直接传 `checkpoint=path` 或 `trainer.resume(path)` 会透明地使用该契约。

## Evaluator

离线评估入口,独立于 `Trainer`;接收模型和 `DataLoader`,复用与训练期相同的 metric 聚合路径,便于固定权重后跑完整 val/test。

::: vgl.engine.Evaluator
    options:
      show_root_heading: true
      show_source: false

## 高级优化器

::: vgl.engine.SAM
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.ASAM
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.GSAM
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.Lookahead
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.ExponentialMovingAverage
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.StochasticWeightAveraging
    options:
      show_root_heading: true
      show_source: false

## 训练调度器

::: vgl.engine.WarmupCosineScheduler
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.GradientAccumulationScheduler
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.GradualUnfreezing
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.LayerwiseLrDecay
    options:
      show_root_heading: true
      show_source: false

## 梯度工具

::: vgl.engine.AdaptiveGradientClipping
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.GradientNoiseInjection
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.GradientValueClipping
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.GradientCentralization
    options:
      show_root_heading: true
      show_source: false

## 损失函数调度器

::: vgl.engine.LabelSmoothingScheduler
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.FocalGammaScheduler
    options:
      show_root_heading: true
      show_source: false

::: vgl.engine.DeferredReweighting
    options:
      show_root_heading: true
      show_source: false
