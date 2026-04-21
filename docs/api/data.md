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
graph = train[0]  # 按需载入当前图
```

### 磁盘布局与读取契约

`OnDiskGraphDataset.write(...)` 会把每张图单独写到 `graphs/graph-*.pt` 下,例如 `graph-000000.pt`:

```text
artifacts/dataset/
├── manifest.json
└── graphs/
    ├── graph-000000.pt
    ├── graph-000001.pt
    └── ...
```

- `OnDiskGraphDataset(root)` 只索引这些文件路径,`dataset[index]` 会按需逐条 `torch.load(...)` 并反序列化当前图。
- `dataset.split(name)` 返回一个 manifest-backed split view,按 `DatasetManifest.splits` 描述的连续区间切出子数据集。
- 新格式支持同构、异构和时序图统一落盘。
- 旧版 `graphs.pt` 数据集仍可读取,因此已有产物不需要迁移即可继续使用。

这种布局把懒加载粒度控制在单图级别:不会在 `OnDiskGraphDataset(...)` 初始化时一次性把整批样本读入内存,但单个样本一旦取出就会还原成普通 `Graph` 对象,后续行为与内存图一致。
