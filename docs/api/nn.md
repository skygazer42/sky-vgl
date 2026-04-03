# vgl.nn

神经网络模块，包含 60+ 图卷积层、Graph Transformer 编码器和池化函数。

## 消息传递基类

::: vgl.nn.MessagePassing
    options:
      show_root_heading: true
      show_source: false

## 同构图卷积层

下表列出所有内置的同构图卷积层：

| 卷积层 | 论文 / 方法 | 说明 |
|--------|-------------|------|
| `GCNConv` | GCN (Kipf & Welling, 2017) | 图卷积网络 |
| `SAGEConv` | GraphSAGE (Hamilton et al., 2017) | 归纳式图学习 |
| `GATConv` | GAT (Veličković et al., 2018) | 图注意力网络 |
| `GATv2Conv` | GATv2 (Brody et al., 2022) | 动态注意力 |
| `GINConv` | GIN (Xu et al., 2019) | 图同构网络 |
| `GraphConv` | GraphConv | 带可选边权的图卷积 |
| `SGConv` | SGC (Wu et al., 2019) | 简化图卷积 |
| `APPNPConv` | APPNP (Gasteiger et al., 2019) | 近似个性化传播 |
| `TAGConv` | TAGCN (Du et al., 2017) | 拓扑自适应图卷积 |
| `ChebConv` | ChebNet (Defferrard et al., 2016) | 切比雪夫图卷积 |
| `AGNNConv` | AGNN (Thekumparampil et al., 2018) | 注意力引导 GNN |
| `LightGCNConv` | LightGCN (He et al., 2020) | 轻量图卷积 |
| `LGConv` | LGC | 线性图卷积 |
| `FAGCNConv` | FAGCN (Bo et al., 2021) | 频率自适应图卷积 |
| `ARMAConv` | ARMA (Bianchi et al., 2021) | ARMA 图卷积 |
| `GPRGNNConv` | GPRGNN (Chien et al., 2021) | 广义 PageRank GNN |
| `MixHopConv` | MixHop (Abu-El-Haija et al., 2019) | 混合跳图卷积 |
| `BernConv` | BernNet (He et al., 2021) | 伯恩斯坦多项式图卷积 |
| `SSGConv` | S²GC (Zhu & Koniusz, 2021) | 简单谱图卷积 |
| `DAGNNConv` | DAGNN (Liu et al., 2020) | 深度自适应图卷积 |
| `GCN2Conv` | GCNII (Chen et al., 2020) | 初始残差图卷积 |
| `H2GCNConv` | H2GCN (Zhu et al., 2020) | 异配图卷积 |
| `EGConv` | EGC (Tailor et al., 2022) | 高效图卷积 |
| `LEConv` | LEConv | 带可学习边权的卷积 |
| `ResGatedGraphConv` | ResGated (Bresson & Laurent, 2018) | 残差门控图卷积 |
| `GatedGraphConv` | GGNN (Li et al., 2016) | 门控图神经网络 |
| `ClusterGCNConv` | ClusterGCN (Chiang et al., 2019) | 集群图卷积 |
| `GENConv` | GENConv (Li et al., 2020) | 图卷积泛化 |
| `FiLMConv` | FiLM (Brockschmidt, 2020) | 特征线性调制卷积 |
| `SimpleConv` | - | 简单消息传递（无参数） |
| `EdgeConv` | EdgeConv (Wang et al., 2019) | 边卷积（DGCNN） |
| `FeaStConv` | FeaSt (Verma et al., 2018) | 特征引导图卷积 |
| `MFConv` | MFConv | 多尺度特征图卷积 |
| `PNAConv` | PNA (Corso et al., 2020) | 主邻域聚合 |
| `GeneralConv` | - | 通用可配置图卷积 |
| `AntiSymmetricConv` | A-DGN (Gravina et al., 2023) | 反对称图卷积 |
| `TransformerConv` | UniMP (Shi et al., 2021) | Transformer 风格图卷积 |
| `WLConvContinuous` | WL (Morris et al., 2019) | 连续 Weisfeiler-Leman |
| `SuperGATConv` | SuperGAT (Kim & Oh, 2022) | 超级图注意力 |
| `DirGNNConv` | Dir-GNN (Rossi et al., 2024) | 方向感知 GNN |
| `CGConv` | CGCNN (Xie & Grossman, 2018) | 晶体图卷积 |
| `DNAConv` | DNA (Fey, 2019) | 动态邻域聚合 |
| `FAConv` | FA (Bo et al., 2021) | 频率自适应卷积 |
| `GatedGCNConv` | GatedGCN | 门控 GCN |
| `PDNConv` | PDN (Rozemberczki et al., 2021) | 路径方向网络 |
| `PointNetConv` | PointNet (Qi et al., 2017) | 点云卷积 |
| `PointTransformerConv` | PointTransformer (Zhao et al., 2021) | 点 Transformer |
| `SplineConv` | SplineCNN (Fey et al., 2018) | 样条卷积 |
| `TWIRLSConv` | TWIRLS (Wang & Zhang, 2022) | 扭转迭代卷积 |

### 各层 API

::: vgl.nn.GCNConv
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.SAGEConv
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.GATConv
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.GINConv
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.GATv2Conv
    options:
      show_root_heading: true
      show_source: false

## 异构图卷积层

| 卷积层 | 说明 |
|--------|------|
| `RGCNConv` | 关系图卷积网络 |
| `RGATConv` | 关系图注意力网络 |
| `HGTConv` | 异构图 Transformer |
| `HANConv` | 异构注意力网络 |
| `HEATConv` | 异构边属性 Transformer |

::: vgl.nn.RGCNConv
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.HGTConv
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.HANConv
    options:
      show_root_heading: true
      show_source: false

## 边特征感知卷积层

| 卷积层 | 说明 |
|--------|------|
| `NNConv` | 边条件化卷积（Neural Network Conv） |
| `ECConv` | 边条件化卷积 |
| `GINEConv` | 带边特征的 GIN |
| `GMMConv` | 高斯混合模型卷积 |

::: vgl.nn.NNConv
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.GINEConv
    options:
      show_root_heading: true
      show_source: false

## Graph Transformer

| 模块 | 说明 |
|------|------|
| `GraphTransformerEncoderLayer` / `GraphTransformerEncoder` | 标准图 Transformer |
| `GraphormerEncoderLayer` / `GraphormerEncoder` | Graphormer |
| `GPSLayer` | GPS (General Powerful Scalable) |
| `NAGphormerEncoder` | NAGphormer |
| `SGFormerEncoderLayer` / `SGFormerEncoder` | SGFormer |

::: vgl.nn.GraphTransformerEncoder
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.GPSLayer
    options:
      show_root_heading: true
      show_source: false

## 时序模块

| 模块 | 说明 |
|------|------|
| `TGNMemory` | TGN 记忆模块 |
| `TGATLayer` / `TGATEncoder` | 时序图注意力 |
| `TimeEncoder` | 时间编码器 |
| `IdentityTemporalMessage` | 恒等时序消息 |
| `LastMessageAggregator` | 最新消息聚合 |
| `MeanMessageAggregator` | 均值消息聚合 |

::: vgl.nn.TGNMemory
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.TimeEncoder
    options:
      show_root_heading: true
      show_source: false

## 辅助模块

### GroupRevRes

::: vgl.nn.GroupRevRes
    options:
      show_root_heading: true
      show_source: false

## 池化函数

```python
from vgl.nn import global_mean_pool, global_sum_pool, global_max_pool
```

::: vgl.nn.global_mean_pool
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.global_sum_pool
    options:
      show_root_heading: true
      show_source: false

::: vgl.nn.global_max_pool
    options:
      show_root_heading: true
      show_source: false
