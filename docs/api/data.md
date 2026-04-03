# vgl.data

数据集模块，包含数据集清单、缓存、内置数据集和磁盘数据集格式。

## 内置数据集

### PlanetoidDataset

::: vgl.data.PlanetoidDataset
    options:
      show_root_heading: true
      show_source: false

支持的数据集：`Cora`、`Citeseer`、`PubMed`。

### TUDataset

::: vgl.data.TUDataset
    options:
      show_root_heading: true
      show_source: false

支持的数据集：`MUTAG`、`PROTEINS`、`ENZYMES`、`IMDB-BINARY` 等标准 TU 图分类集合。

### KarateClubDataset

::: vgl.data.KarateClubDataset
    options:
      show_root_heading: true
      show_source: false

零下载入门数据集。

## 数据集注册表

### DatasetRegistry

::: vgl.data.DatasetRegistry
    options:
      show_root_heading: true
      show_source: false

支持的别名格式：

- 简写：`cora`、`citeseer`、`mutag`、`proteins`
- 变体：`karate_club`、`karateclub`、`imdb_binary`、`imdbbinary`
- 前缀：`planetoid:cora`、`tu:proteins`、`tu/proteins`、`tu.proteins`

## 数据集清单

### DatasetManifest

::: vgl.data.DatasetManifest
    options:
      show_root_heading: true
      show_source: false

### DatasetSplit

::: vgl.data.DatasetSplit
    options:
      show_root_heading: true
      show_source: false

## 磁盘数据集

### OnDiskGraphDataset

::: vgl.data.OnDiskGraphDataset
    options:
      show_root_heading: true
      show_source: false

```python
from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.data.ondisk import OnDiskGraphDataset

manifest = DatasetManifest(
    name="my-dataset",
    version="1.0",
    splits=(DatasetSplit("train", size=100), DatasetSplit("test", size=20)),
)
OnDiskGraphDataset.write("artifacts/dataset", manifest, graphs)

dataset = OnDiskGraphDataset("artifacts/dataset")
train = dataset.split("train")
graph = train[0]  # 延迟加载
```
