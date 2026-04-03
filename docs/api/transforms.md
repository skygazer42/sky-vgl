# vgl.transforms

数据变换模块，用于图数据的预处理和增强。

## Compose

::: vgl.transforms.Compose
    options:
      show_root_heading: true
      show_source: false

## 特征变换

::: vgl.transforms.NormalizeFeatures
    options:
      show_root_heading: true
      show_source: false

::: vgl.transforms.FeatureStandardize
    options:
      show_root_heading: true
      show_source: false

::: vgl.transforms.TrainOnlyFeatureNormalizer
    options:
      show_root_heading: true
      show_source: false

## 数据集切分

::: vgl.transforms.RandomNodeSplit
    options:
      show_root_heading: true
      show_source: false

::: vgl.transforms.RandomGraphSplit
    options:
      show_root_heading: true
      show_source: false

::: vgl.transforms.RandomLinkSplit
    options:
      show_root_heading: true
      show_source: false

## 结构变换

::: vgl.transforms.ToUndirected
    options:
      show_root_heading: true
      show_source: false

::: vgl.transforms.AddSelfLoops
    options:
      show_root_heading: true
      show_source: false

::: vgl.transforms.RemoveSelfLoops
    options:
      show_root_heading: true
      show_source: false

::: vgl.transforms.LargestConnectedComponents
    options:
      show_root_heading: true
      show_source: false
