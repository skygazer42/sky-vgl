# 卷积层速查

VGL 内置 60+ 种图卷积层，覆盖同构、异构、边特征、时序等场景。

## 同构图卷积

### 基础卷积

| 层 | 导入 | 关键参数 | 适用场景 |
|----|------|----------|----------|
| `GCNConv` | `from vgl.nn import GCNConv` | `in_channels, out_channels` | 通用节点分类基线 |
| `SAGEConv` | `from vgl.nn import SAGEConv` | `in_channels, out_channels` | 归纳式学习 |
| `GATConv` | `from vgl.nn import GATConv` | `in_channels, out_channels, num_heads` | 需要注意力权重 |
| `GATv2Conv` | `from vgl.nn import GATv2Conv` | `in_channels, out_channels, num_heads` | 动态注意力 |
| `GINConv` | `from vgl.nn import GINConv` | `nn (MLP)` | 图分类（强表达力） |
| `GraphConv` | `from vgl.nn import GraphConv` | `in_channels, out_channels` | 通用（带可选边权） |

### 谱方法

| 层 | 导入 | 关键参数 | 说明 |
|----|------|----------|------|
| `ChebConv` | `from vgl.nn import ChebConv` | `in_channels, out_channels, K` | K 阶切比雪夫多项式 |
| `BernConv` | `from vgl.nn import BernConv` | `in_channels, out_channels, K` | 伯恩斯坦多项式 |
| `ARMAConv` | `from vgl.nn import ARMAConv` | `in_channels, out_channels` | ARMA 滤波器 |

### 简化 / 高效模型

| 层 | 导入 | 说明 |
|----|------|------|
| `SGConv` | `from vgl.nn import SGConv` | 简化图卷积（一步 K 跳） |
| `SSGConv` | `from vgl.nn import SSGConv` | 简单谱图卷积 |
| `LightGCNConv` | `from vgl.nn import LightGCNConv` | 推荐系统轻量卷积 |
| `LGConv` | `from vgl.nn import LGConv` | 线性图卷积 |
| `SimpleConv` | `from vgl.nn import SimpleConv` | 无参数消息传递 |
| `EGConv` | `from vgl.nn import EGConv` | 高效图卷积 |

### 深层 GNN

| 层 | 导入 | 说明 |
|----|------|------|
| `APPNPConv` | `from vgl.nn import APPNPConv` | 近似个性化传播 |
| `GCN2Conv` | `from vgl.nn import GCN2Conv` | GCNII（初始残差） |
| `DAGNNConv` | `from vgl.nn import DAGNNConv` | 深度自适应 |
| `GPRGNNConv` | `from vgl.nn import GPRGNNConv` | 广义 PageRank |
| `H2GCNConv` | `from vgl.nn import H2GCNConv` | 异配图深层卷积 |
| `FAGCNConv` | `from vgl.nn import FAGCNConv` | 频率自适应 |
| `TAGConv` | `from vgl.nn import TAGConv` | 拓扑自适应 |
| `MixHopConv` | `from vgl.nn import MixHopConv` | 混合跳 |

### 门控 / 残差

| 层 | 导入 | 说明 |
|----|------|------|
| `ResGatedGraphConv` | `from vgl.nn import ResGatedGraphConv` | 残差门控 |
| `GatedGraphConv` | `from vgl.nn import GatedGraphConv` | GGNN |
| `GatedGCNConv` | `from vgl.nn import GatedGCNConv` | 门控 GCN |
| `AGNNConv` | `from vgl.nn import AGNNConv` | 注意力引导 |

### 高级 / 其他

| 层 | 导入 | 说明 |
|----|------|------|
| `PNAConv` | `from vgl.nn import PNAConv` | 主邻域聚合 |
| `GENConv` | `from vgl.nn import GENConv` | 泛化卷积 |
| `FiLMConv` | `from vgl.nn import FiLMConv` | 特征线性调制 |
| `ClusterGCNConv` | `from vgl.nn import ClusterGCNConv` | Cluster-GCN |
| `TransformerConv` | `from vgl.nn import TransformerConv` | Transformer 风格 |
| `SuperGATConv` | `from vgl.nn import SuperGATConv` | 超级 GAT |
| `DirGNNConv` | `from vgl.nn import DirGNNConv` | 方向感知 |
| `AntiSymmetricConv` | `from vgl.nn import AntiSymmetricConv` | 反对称 |
| `WLConvContinuous` | `from vgl.nn import WLConvContinuous` | 连续 WL |
| `GeneralConv` | `from vgl.nn import GeneralConv` | 通用可配置 |
| `EdgeConv` | `from vgl.nn import EdgeConv` | DGCNN 边卷积 |
| `FeaStConv` | `from vgl.nn import FeaStConv` | 特征引导 |
| `MFConv` | `from vgl.nn import MFConv` | 多尺度特征 |
| `LEConv` | `from vgl.nn import LEConv` | 可学习边权 |
| `FAConv` | `from vgl.nn import FAConv` | 频率自适应 |
| `DNAConv` | `from vgl.nn import DNAConv` | 动态邻域聚合 |
| `PDNConv` | `from vgl.nn import PDNConv` | 路径方向网络 |
| `TWIRLSConv` | `from vgl.nn import TWIRLSConv` | 扭转迭代 |
| `CGConv` | `from vgl.nn import CGConv` | 晶体图 |
| `SplineConv` | `from vgl.nn import SplineConv` | 样条卷积 |
| `PointNetConv` | `from vgl.nn import PointNetConv` | 点云 |
| `PointTransformerConv` | `from vgl.nn import PointTransformerConv` | 点 Transformer |

## 异构图卷积

| 层 | 导入 | 说明 |
|----|------|------|
| `RGCNConv` | `from vgl.nn import RGCNConv` | 关系图卷积 |
| `RGATConv` | `from vgl.nn import RGATConv` | 关系图注意力 |
| `HGTConv` | `from vgl.nn import HGTConv` | 异构图 Transformer |
| `HANConv` | `from vgl.nn import HANConv` | 异构注意力网络 |
| `HEATConv` | `from vgl.nn import HEATConv` | 异构边属性 Transformer |

## 边特征感知卷积

| 层 | 导入 | 说明 |
|----|------|------|
| `NNConv` | `from vgl.nn import NNConv` | 神经网络边条件化 |
| `ECConv` | `from vgl.nn import ECConv` | 边条件化 |
| `GINEConv` | `from vgl.nn import GINEConv` | GIN + 边特征 |
| `GMMConv` | `from vgl.nn import GMMConv` | 高斯混合模型 |

## Graph Transformer

| 模块 | 导入 | 说明 |
|------|------|------|
| `GraphTransformerEncoder` | `from vgl.nn import GraphTransformerEncoder` | 标准图 Transformer |
| `GraphormerEncoder` | `from vgl.nn import GraphormerEncoder` | Graphormer |
| `GPSLayer` | `from vgl.nn import GPSLayer` | GPS |
| `NAGphormerEncoder` | `from vgl.nn import NAGphormerEncoder` | NAGphormer |
| `SGFormerEncoder` | `from vgl.nn import SGFormerEncoder` | SGFormer |

## 辅助模块

| 模块 | 导入 | 说明 |
|------|------|------|
| `GroupRevRes` | `from vgl.nn import GroupRevRes` | 分组可逆残差包装器 |
| `global_mean_pool` | `from vgl.nn import global_mean_pool` | 均值池化 |
| `global_sum_pool` | `from vgl.nn import global_sum_pool` | 求和池化 |
| `global_max_pool` | `from vgl.nn import global_max_pool` | 最大值池化 |
