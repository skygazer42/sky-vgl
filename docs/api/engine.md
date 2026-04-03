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
