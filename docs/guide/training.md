# 训练器与回调

`Trainer` 是 VGL 的训练引擎，负责训练循环、验证、测试、checkpoint 保存和日志记录。

## 基本使用

```python
from vgl.engine import Trainer

trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=200,
)

history = trainer.fit(graph, val_data=graph)
test_result = trainer.test(graph)
```

## 监控和 Checkpoint

```python
trainer = Trainer(
    model=model,
    task=task,
    optimizer=torch.optim.Adam,
    lr=1e-3,
    max_epochs=200,
    monitor="val_accuracy",          # 监控指标
    save_best_path="best.pt",       # 保存最佳模型
)
```

加载 checkpoint：

```python
from vgl.engine import load_checkpoint, restore_checkpoint

state = load_checkpoint("best.pt")
model = restore_checkpoint(model, "best.pt")
```

## Logger

VGL 内置四种 Logger：

### ConsoleLogger（默认启用）

```python
trainer = Trainer(
    ...,
    console_mode="detailed",        # "compact" 或 "detailed"
    console_theme="cat",            # ASCII 猫咪状态主题
    console_metric_names={"loss", "val_loss"},
    console_show_learning_rate=False,
    console_show_events=False,
    console_show_timestamp=False,
    enable_progress_bar=False,
)
```

### JSONLinesLogger

```python
from vgl.engine import JSONLinesLogger

logger = JSONLinesLogger(
    "artifacts/train.jsonl",
    events={"epoch_end", "fit_end"},     # 事件过滤
    metric_names={"train_loss", "val_loss"},  # 指标过滤
    include_context=False,
    show_learning_rate=False,
    flush=True,
)
```

### CSVLogger

```python
from vgl.engine import CSVLogger

logger = CSVLogger(
    "artifacts/epochs.csv",
    events={"epoch_end"},
    metric_names={"train_loss", "val_loss"},
)
```

### TensorBoardLogger

```python
from vgl.engine import TensorBoardLogger

logger = TensorBoardLogger(
    "artifacts/tensorboard",
    events={"train_step", "epoch_end"},
    flush=True,
)
# tensorboard --logdir artifacts/tensorboard
```

使用多个 Logger：

```python
trainer = Trainer(
    ...,
    loggers=[
        JSONLinesLogger("artifacts/train.jsonl", flush=True),
        TensorBoardLogger("artifacts/tb"),
    ],
    log_every_n_steps=10,
)
```

## Callback

### EarlyStopping

```python
from vgl.engine import EarlyStopping

trainer = Trainer(
    ...,
    callbacks=[
        EarlyStopping(patience=20, monitor="val_accuracy"),
    ],
)
```

### HistoryLogger

```python
from vgl.engine import HistoryLogger

history_logger = HistoryLogger()
trainer = Trainer(
    ...,
    callbacks=[history_logger],
)
# 训练后通过 history_logger 访问完整历史
```

### 自定义 Callback

```python
from vgl.engine import Callback, StopTraining

class MyCallback(Callback):
    def on_fit_start(self, trainer, **kwargs):
        print("训练开始")

    def on_epoch_end(self, trainer, **kwargs):
        # 可以访问 TrainingHistory
        if some_condition:
            raise StopTraining("自定义停止条件")

    def on_fit_end(self, trainer, **kwargs):
        print("训练结束")
```

## 调试和实验

```python
trainer = Trainer(
    ...,
    default_root_dir="artifacts/debug-run",
    run_name="sanity-check",
    fast_dev_run=True,               # 快速验证流程
    num_sanity_val_steps=2,          # 训练前运行验证
    val_check_interval=0.5,          # epoch 中期验证
    profiler="simple",               # 耗时统计
)
```

批次限制（不启用 fast_dev_run）：

```python
trainer = Trainer(
    ...,
    limit_train_batches=10,
    limit_val_batches=5,
    limit_test_batches=5,
)
```

### Runtime Reliability Tips

Use the queued torch.compile-based smoke test (see `tests/test_runtime_compile_smoke.py`) when PyTorch 2.4+ is available to verify the same `Graph`/`Trainer` loop supports Dynamo-backed execution without extra hooks. The release profiler already records per-fit and per-epoch timing under the `"profile"` field, so nightly runs that enable `profiler="simple"` can assert these keys remain present across releases.

The benchmark artifact emitted by `scripts/benchmark_hotpaths.py` is now treated as a versioned contract. Downstream consumers can rely on these top-level fields:

```bash
python scripts/benchmark_hotpaths.py --preset ci --output artifacts/benchmark-hotpaths.json
```

- `schema_version`
- `benchmark`
- `generated_at_utc`
- `preset`
- `config`
- `runner`
- `metric_unit`
- `query_ops`
- `routing`
- `sampling`

All timing metrics are reported in seconds under `metric_unit`, and the canonical CI artifact path is `artifacts/benchmark-hotpaths.json`. The CI benchmark job also prints the JSON and uploads the artifact directly so schema drift and performance drift can be reviewed together.

## TrainingHistory

`trainer.fit()` 返回 `TrainingHistory` 对象：

```python
history = trainer.fit(graph, val_data=graph)
# history 包含 epoch 摘要、监控元数据、耗时和早停状态
```

## 结构化生命周期事件

Logger 记录的结构化事件包括：

| 事件 | 说明 |
|------|------|
| `fit_start` | 训练开始，包含运行元数据 |
| `stage_start` | 每个 train / val / test / sanity_val 阶段开始，包含 `total_batches` |
| `train_step` | 每个训练步骤 |
| `epoch_end` | 每个 epoch 结束 |
| `evaluate_end` | 每个验证或测试阶段结束 |
| `monitor_improved` | 监控指标改善，包含改善量 |
| `checkpoint_saved` | checkpoint 保存，包含文件大小和耗时 |
| `exception` | 训练过程中的结构化异常记录 |
| `fit_end` | 训练结束 |

无上下文 logger 仍然会保留每个事件的核心字段。例如 `stage_start` 至少保留 `event / stage / epoch / epochs / global_step / batch_idx / total_batches`，而 `checkpoint_saved` 还会保留 `checkpoint_tag / path / size_bytes / save_seconds / format / format_version`。这让 JSONL / CSV 过滤模式在裁剪上下文后仍然保持可机器消费的事件合同。

## 高级训练工具

VGL 还提供多种高级训练回调和优化器包装：

| 工具 | 说明 |
|------|------|
| `SAM` / `ASAM` / `GSAM` | Sharpness-Aware Minimization |
| `Lookahead` | Lookahead 优化器包装 |
| `ExponentialMovingAverage` | 模型 EMA |
| `StochasticWeightAveraging` | 随机权重平均 |
| `GradualUnfreezing` | 渐进式解冻 |
| `GradientAccumulationScheduler` | 梯度累积调度 |
| `GradientNoiseInjection` | 梯度噪声注入 |
| `AdaptiveGradientClipping` | 自适应梯度裁剪 |
| `WarmupCosineScheduler` | Warmup + Cosine 学习率 |
| `LayerwiseLrDecay` | 分层学习率衰减 |

## 下一步

- [数据变换](transforms.md) — 数据预处理
- [API 参考: vgl.engine](../api/engine.md) — Trainer 完整 API
