# Stable root exports: canonical quickstart imports that new docs and examples may rely on.
from vgl.graph import Graph as Graph
from vgl.graph import GraphBatch as GraphBatch
from vgl.dataloading import DataLoader as DataLoader
from vgl.engine import Trainer as Trainer
from vgl.nn import MessagePassing as MessagePassing
from vgl.tasks import NodeClassificationTask as NodeClassificationTask
from vgl.tasks import GraphClassificationTask as GraphClassificationTask
from vgl.tasks import LinkPredictionTask as LinkPredictionTask
from vgl.tasks import TemporalEventPredictionTask as TemporalEventPredictionTask
from vgl.version import __version__ as __version__
from vgl.data import DatasetRegistry as DatasetRegistry
from vgl.data import KarateClubDataset as KarateClubDataset
from vgl.data import PlanetoidDataset as PlanetoidDataset
from vgl.data import TUDataset as TUDataset
# Compatibility-only root exports: supported for existing code, but new imports should prefer
# the owning domain modules directly.
from vgl.nn import AGNNConv as AGNNConv
from vgl.nn import APPNPConv as APPNPConv
from vgl.nn import ARMAConv as ARMAConv
from vgl.engine import ASAM as ASAM
from vgl.metrics import Accuracy as Accuracy
from vgl.engine import AdaptiveGradientClipping as AdaptiveGradientClipping
from vgl.nn import AntiSymmetricConv as AntiSymmetricConv
from vgl.nn import BernConv as BernConv
from vgl.graph import Block as Block
from vgl.engine import BootstrapBetaScheduler as BootstrapBetaScheduler
from vgl.tasks import BootstrapTask as BootstrapTask
from vgl.nn import CGConv as CGConv
from vgl.engine import CSVLogger as CSVLogger
from vgl.engine import Callback as Callback
from vgl.dataloading import CandidateLinkSampler as CandidateLinkSampler
from vgl.nn import ChebConv as ChebConv
from vgl.dataloading import ClusterData as ClusterData
from vgl.nn import ClusterGCNConv as ClusterGCNConv
from vgl.dataloading import ClusterLoader as ClusterLoader
from vgl.engine import ConfidencePenaltyScheduler as ConfidencePenaltyScheduler
from vgl.tasks import ConfidencePenaltyTask as ConfidencePenaltyTask
from vgl.engine import ConsoleLogger as ConsoleLogger
from vgl.nn import DAGNNConv as DAGNNConv
from vgl.nn import DNAConv as DNAConv
from vgl.data import DatasetManifest as DatasetManifest
from vgl.data import DatasetSplit as DatasetSplit
from vgl.engine import DeferredReweighting as DeferredReweighting
from vgl.nn import DirGNNConv as DirGNNConv
from vgl.nn import ECConv as ECConv
from vgl.nn import EGConv as EGConv
from vgl.engine import EarlyStopping as EarlyStopping
from vgl.nn import EdgeConv as EdgeConv
from vgl.engine import ExponentialMovingAverage as ExponentialMovingAverage
from vgl.nn import FAConv as FAConv
from vgl.nn import FAGCNConv as FAGCNConv
from vgl.nn import FeaStConv as FeaStConv
from vgl.nn import FiLMConv as FiLMConv
from vgl.metrics import FilteredHitsAtK as FilteredHitsAtK
from vgl.metrics import FilteredMRR as FilteredMRR
from vgl.engine import FloodingLevelScheduler as FloodingLevelScheduler
from vgl.tasks import FloodingTask as FloodingTask
from vgl.engine import FocalGammaScheduler as FocalGammaScheduler
from vgl.dataloading import FullGraphSampler as FullGraphSampler
from vgl.nn import GATConv as GATConv
from vgl.nn import GATv2Conv as GATv2Conv
from vgl.nn import GCN2Conv as GCN2Conv
from vgl.nn import GCNConv as GCNConv
from vgl.nn import GENConv as GENConv
from vgl.nn import GINConv as GINConv
from vgl.nn import GINEConv as GINEConv
from vgl.nn import GMMConv as GMMConv
from vgl.nn import GPRGNNConv as GPRGNNConv
from vgl.nn import GPSLayer as GPSLayer
from vgl.engine import GSAM as GSAM
from vgl.nn import GatedGCNConv as GatedGCNConv
from vgl.nn import GatedGraphConv as GatedGraphConv
from vgl.nn import GeneralConv as GeneralConv
from vgl.engine import GeneralizedCrossEntropyScheduler as GeneralizedCrossEntropyScheduler
from vgl.tasks import GeneralizedCrossEntropyTask as GeneralizedCrossEntropyTask
from vgl.engine import GradientAccumulationScheduler as GradientAccumulationScheduler
from vgl.engine import GradientCentralization as GradientCentralization
from vgl.engine import GradientNoiseInjection as GradientNoiseInjection
from vgl.engine import GradientValueClipping as GradientValueClipping
from vgl.engine import GradualUnfreezing as GradualUnfreezing
from vgl.nn import GraphConv as GraphConv
from vgl.dataloading import GraphSAINTEdgeSampler as GraphSAINTEdgeSampler
from vgl.dataloading import GraphSAINTNodeSampler as GraphSAINTNodeSampler
from vgl.dataloading import GraphSAINTRandomWalkSampler as GraphSAINTRandomWalkSampler
from vgl.graph import GraphSchema as GraphSchema
from vgl.nn import GraphTransformerEncoder as GraphTransformerEncoder
from vgl.nn import GraphTransformerEncoderLayer as GraphTransformerEncoderLayer
from vgl.graph import GraphView as GraphView
from vgl.nn import GraphormerEncoder as GraphormerEncoder
from vgl.nn import GraphormerEncoderLayer as GraphormerEncoderLayer
from vgl.nn import GroupRevRes as GroupRevRes
from vgl.nn import H2GCNConv as H2GCNConv
from vgl.nn import HANConv as HANConv
from vgl.nn import HEATConv as HEATConv
from vgl.nn import HGTConv as HGTConv
from vgl.dataloading import HardNegativeLinkSampler as HardNegativeLinkSampler
from vgl.graph import HeteroBlock as HeteroBlock
from vgl.engine import HistoryLogger as HistoryLogger
from vgl.metrics import HitsAtK as HitsAtK
from vgl.nn import IdentityTemporalMessage as IdentityTemporalMessage
from vgl.engine import JSONLinesLogger as JSONLinesLogger
from vgl.nn import LEConv as LEConv
from vgl.nn import LGConv as LGConv
from vgl.engine import LabelSmoothingScheduler as LabelSmoothingScheduler
from vgl.nn import LastMessageAggregator as LastMessageAggregator
from vgl.engine import LayerwiseLrDecay as LayerwiseLrDecay
from vgl.engine import LdamMarginScheduler as LdamMarginScheduler
from vgl.nn import LightGCNConv as LightGCNConv
from vgl.dataloading import LinkNeighborSampler as LinkNeighborSampler
from vgl.graph import LinkPredictionBatch as LinkPredictionBatch
from vgl.dataloading import LinkPredictionRecord as LinkPredictionRecord
from vgl.dataloading import ListDataset as ListDataset
from vgl.dataloading import Loader as Loader
from vgl.engine import Logger as Logger
from vgl.engine import LogitAdjustTauScheduler as LogitAdjustTauScheduler
from vgl.engine import Lookahead as Lookahead
from vgl.nn import MFConv as MFConv
from vgl.metrics import MRR as MRR
from vgl.nn import MeanMessageAggregator as MeanMessageAggregator
from vgl.metrics import Metric as Metric
from vgl.nn import MixHopConv as MixHopConv
from vgl.engine import ModelCheckpoint as ModelCheckpoint
from vgl.nn import NAGphormerEncoder as NAGphormerEncoder
from vgl.nn import NNConv as NNConv
from vgl.dataloading import Node2VecWalkSampler as Node2VecWalkSampler
from vgl.graph import NodeBatch as NodeBatch
from vgl.dataloading import NodeNeighborSampler as NodeNeighborSampler
from vgl.dataloading import NodeSeedSubgraphSampler as NodeSeedSubgraphSampler
from vgl.data import OnDiskGraphDataset as OnDiskGraphDataset
from vgl.nn import PDNConv as PDNConv
from vgl.nn import PNAConv as PNAConv
from vgl.nn import PointNetConv as PointNetConv
from vgl.nn import PointTransformerConv as PointTransformerConv
from vgl.tasks import Poly1CrossEntropyTask as Poly1CrossEntropyTask
from vgl.engine import Poly1EpsilonScheduler as Poly1EpsilonScheduler
from vgl.engine import PosWeightScheduler as PosWeightScheduler
from vgl.tasks import RDropTask as RDropTask
from vgl.nn import RGATConv as RGATConv
from vgl.nn import RGCNConv as RGCNConv
from vgl.dataloading import RandomWalkSampler as RandomWalkSampler
from vgl.nn import ResGatedGraphConv as ResGatedGraphConv
from vgl.nn import SAGEConv as SAGEConv
from vgl.engine import SAM as SAM
from vgl.nn import SGConv as SGConv
from vgl.nn import SGFormerEncoder as SGFormerEncoder
from vgl.nn import SGFormerEncoderLayer as SGFormerEncoderLayer
from vgl.nn import SSGConv as SSGConv
from vgl.dataloading import SampleRecord as SampleRecord
from vgl.dataloading import ShaDowKHopSampler as ShaDowKHopSampler
from vgl.nn import SimpleConv as SimpleConv
from vgl.nn import SplineConv as SplineConv
from vgl.engine import StochasticWeightAveraging as StochasticWeightAveraging
from vgl.engine import StopTraining as StopTraining
from vgl.nn import SuperGATConv as SuperGATConv
from vgl.engine import SymmetricCrossEntropyBetaScheduler as SymmetricCrossEntropyBetaScheduler
from vgl.tasks import SymmetricCrossEntropyTask as SymmetricCrossEntropyTask
from vgl.nn import TAGConv as TAGConv
from vgl.nn import TGATEncoder as TGATEncoder
from vgl.nn import TGATLayer as TGATLayer
from vgl.nn import TGNMemory as TGNMemory
from vgl.nn import TWIRLSConv as TWIRLSConv
from vgl.tasks import Task as Task
from vgl.graph import TemporalEventBatch as TemporalEventBatch
from vgl.dataloading import TemporalEventRecord as TemporalEventRecord
from vgl.dataloading import TemporalNeighborSampler as TemporalNeighborSampler
from vgl.engine import TensorBoardLogger as TensorBoardLogger
from vgl.nn import TimeEncoder as TimeEncoder
from vgl.engine import TrainingHistory as TrainingHistory
from vgl.nn import TransformerConv as TransformerConv
from vgl.dataloading import UniformNegativeLinkSampler as UniformNegativeLinkSampler
from vgl.nn import WLConvContinuous as WLConvContinuous
from vgl.engine import WarmupCosineScheduler as WarmupCosineScheduler
from vgl.engine import WeightDecayScheduler as WeightDecayScheduler
from vgl.nn import global_max_pool as global_max_pool
from vgl.nn import global_mean_pool as global_mean_pool
from vgl.nn import global_sum_pool as global_sum_pool

# Keep stable exports first, followed by compatibility-only exports in a
# deterministic alphabetical order.
_ROOT_STABLE_EXPORTS = (
    "Graph",
    "GraphBatch",
    "DataLoader",
    "Trainer",
    "MessagePassing",
    "NodeClassificationTask",
    "GraphClassificationTask",
    "LinkPredictionTask",
    "TemporalEventPredictionTask",
    "__version__",
    "DatasetRegistry",
    "KarateClubDataset",
    "PlanetoidDataset",
    "TUDataset",
)

_ROOT_COMPATIBILITY_EXPORTS = tuple(
    sorted(
        (
            "AGNNConv",
            "APPNPConv",
            "ARMAConv",
            "ASAM",
            "Accuracy",
            "AdaptiveGradientClipping",
            "AntiSymmetricConv",
            "BernConv",
            "Block",
            "BootstrapBetaScheduler",
            "BootstrapTask",
            "CGConv",
            "CSVLogger",
            "Callback",
            "CandidateLinkSampler",
            "ChebConv",
            "ClusterData",
            "ClusterGCNConv",
            "ClusterLoader",
            "ConfidencePenaltyScheduler",
            "ConfidencePenaltyTask",
            "ConsoleLogger",
            "DAGNNConv",
            "DNAConv",
            "DatasetManifest",
            "DatasetSplit",
            "DeferredReweighting",
            "DirGNNConv",
            "ECConv",
            "EGConv",
            "EarlyStopping",
            "EdgeConv",
            "ExponentialMovingAverage",
            "FAConv",
            "FAGCNConv",
            "FeaStConv",
            "FiLMConv",
            "FilteredHitsAtK",
            "FilteredMRR",
            "FloodingLevelScheduler",
            "FloodingTask",
            "FocalGammaScheduler",
            "FullGraphSampler",
            "GATConv",
            "GATv2Conv",
            "GCN2Conv",
            "GCNConv",
            "GENConv",
            "GINConv",
            "GINEConv",
            "GMMConv",
            "GPRGNNConv",
            "GPSLayer",
            "GSAM",
            "GatedGCNConv",
            "GatedGraphConv",
            "GeneralConv",
            "GeneralizedCrossEntropyScheduler",
            "GeneralizedCrossEntropyTask",
            "GradientAccumulationScheduler",
            "GradientCentralization",
            "GradientNoiseInjection",
            "GradientValueClipping",
            "GradualUnfreezing",
            "GraphConv",
            "GraphSAINTEdgeSampler",
            "GraphSAINTNodeSampler",
            "GraphSAINTRandomWalkSampler",
            "GraphSchema",
            "GraphTransformerEncoder",
            "GraphTransformerEncoderLayer",
            "GraphView",
            "GraphormerEncoder",
            "GraphormerEncoderLayer",
            "GroupRevRes",
            "H2GCNConv",
            "HANConv",
            "HEATConv",
            "HGTConv",
            "HardNegativeLinkSampler",
            "HeteroBlock",
            "HistoryLogger",
            "HitsAtK",
            "IdentityTemporalMessage",
            "JSONLinesLogger",
            "LEConv",
            "LGConv",
            "LabelSmoothingScheduler",
            "LastMessageAggregator",
            "LayerwiseLrDecay",
            "LdamMarginScheduler",
            "LightGCNConv",
            "LinkNeighborSampler",
            "LinkPredictionBatch",
            "LinkPredictionRecord",
            "ListDataset",
            "Loader",
            "Logger",
            "LogitAdjustTauScheduler",
            "Lookahead",
            "MFConv",
            "MRR",
            "MeanMessageAggregator",
            "Metric",
            "MixHopConv",
            "ModelCheckpoint",
            "NAGphormerEncoder",
            "NNConv",
            "Node2VecWalkSampler",
            "NodeBatch",
            "NodeNeighborSampler",
            "NodeSeedSubgraphSampler",
            "OnDiskGraphDataset",
            "PDNConv",
            "PNAConv",
            "PointNetConv",
            "PointTransformerConv",
            "Poly1CrossEntropyTask",
            "Poly1EpsilonScheduler",
            "PosWeightScheduler",
            "RDropTask",
            "RGATConv",
            "RGCNConv",
            "RandomWalkSampler",
            "ResGatedGraphConv",
            "SAGEConv",
            "SAM",
            "SGConv",
            "SGFormerEncoder",
            "SGFormerEncoderLayer",
            "SSGConv",
            "SampleRecord",
            "ShaDowKHopSampler",
            "SimpleConv",
            "SplineConv",
            "StochasticWeightAveraging",
            "StopTraining",
            "SuperGATConv",
            "SymmetricCrossEntropyBetaScheduler",
            "SymmetricCrossEntropyTask",
            "TAGConv",
            "TGATEncoder",
            "TGATLayer",
            "TGNMemory",
            "TWIRLSConv",
            "Task",
            "TemporalEventBatch",
            "TemporalEventRecord",
            "TemporalNeighborSampler",
            "TensorBoardLogger",
            "TimeEncoder",
            "TrainingHistory",
            "TransformerConv",
            "UniformNegativeLinkSampler",
            "WLConvContinuous",
            "WarmupCosineScheduler",
            "WeightDecayScheduler",
            "global_max_pool",
            "global_mean_pool",
            "global_sum_pool",
        )
    )
)

__all__ = [*_ROOT_STABLE_EXPORTS, *_ROOT_COMPATIBILITY_EXPORTS]
