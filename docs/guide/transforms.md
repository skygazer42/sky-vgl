# 数据变换

VGL 提供一系列数据变换（transforms），用于图数据的预处理和增强。

## Compose

使用 `Compose` 将多个变换串联：

```python
from vgl.transforms import Compose, NormalizeFeatures, RandomNodeSplit

transform = Compose([
    NormalizeFeatures(),
    RandomNodeSplit(num_train_per_class=20, num_val=500, num_test=1000),
])
```

## 特征变换

### NormalizeFeatures

按行归一化节点特征（L1 归一化）：

```python
from vgl.transforms import NormalizeFeatures

transform = NormalizeFeatures()
```

### FeatureStandardize

对特征进行标准化（零均值、单位方差）：

```python
from vgl.transforms import FeatureStandardize

transform = FeatureStandardize()
```

### TrainOnlyFeatureNormalizer

仅在训练集上计算归一化统计量：

```python
from vgl.transforms import TrainOnlyFeatureNormalizer

transform = TrainOnlyFeatureNormalizer()
```

## 数据集切分

### RandomNodeSplit

为节点分类创建训练/验证/测试掩码：

```python
from vgl.transforms import RandomNodeSplit

transform = RandomNodeSplit(
    num_train_per_class=20,
    num_val=500,
    num_test=1000,
    seed=42,
)
```

### RandomGraphSplit

为图分类数据集创建训练/验证/测试切分：

```python
from vgl.transforms import RandomGraphSplit

transform = RandomGraphSplit(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
)
```

### RandomLinkSplit

为链接预测创建边级别切分，自动生成负样本：

```python
from vgl.transforms import RandomLinkSplit

# 自动创建 LinkPredictionRecord 数据集
```

## 结构变换

### ToUndirected

将有向图转换为无向图（添加反向边）：

```python
from vgl.transforms import ToUndirected

transform = ToUndirected()
```

### AddSelfLoops

为每个节点添加自环：

```python
from vgl.transforms import AddSelfLoops

transform = AddSelfLoops()
```

### RemoveSelfLoops

移除图中的自环：

```python
from vgl.transforms import RemoveSelfLoops

transform = RemoveSelfLoops()
```

### LargestConnectedComponents

保留最大连通分量：

```python
from vgl.transforms import LargestConnectedComponents

transform = LargestConnectedComponents()
```

## 完整变换列表

| 变换 | 说明 |
|------|------|
| `Compose` | 串联多个变换 |
| `NormalizeFeatures` | L1 行归一化 |
| `FeatureStandardize` | 零均值单位方差标准化 |
| `TrainOnlyFeatureNormalizer` | 仅训练集统计量归一化 |
| `RandomNodeSplit` | 节点级随机切分 |
| `RandomGraphSplit` | 图级随机切分 |
| `RandomLinkSplit` | 边级随机切分（链接预测） |
| `ToUndirected` | 转为无向图 |
| `AddSelfLoops` | 添加自环 |
| `RemoveSelfLoops` | 移除自环 |
| `LargestConnectedComponents` | 保留最大连通分量 |

## 与数据集配合

变换通常在加载数据集时应用：

```python
from vgl.data import PlanetoidDataset
from vgl.transforms import Compose, NormalizeFeatures

dataset = PlanetoidDataset(
    root="data",
    name="Cora",
    transform=Compose([NormalizeFeatures()]),
)
```

## 下一步

- [快速入门](../getting-started/quickstart.md) — 变换的实际应用
- [API 参考: vgl.transforms](../api/transforms.md) — 完整 API
