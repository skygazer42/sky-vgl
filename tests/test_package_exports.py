from vgl import (
    AGNNConv,
    ASAM,
    Accuracy,
    AdaptiveGradientClipping,
    AntiSymmetricConv,
    APPNPConv,
    ARMAConv,
    BernConv,
    BootstrapBetaScheduler,
    BootstrapTask,
    CSVLogger,
    CGConv,
    ChebConv,
    ClusterData,
    ClusterGCNConv,
    ClusterLoader,
    ConsoleLogger,
    ConfidencePenaltyScheduler,
    ConfidencePenaltyTask,
    DatasetRegistry,
    DAGNNConv,
    DNAConv,
    EdgeConv,
    EGConv,
    FAConv,
    FAGCNConv,
    FiLMConv,
    FeaStConv,
    FocalGammaScheduler,
    FloodingLevelScheduler,
    GeneralizedCrossEntropyScheduler,
    GradientAccumulationScheduler,
    GradientNoiseInjection,
    GradientValueClipping,
    GATConv,
    FilteredHitsAtK,
    FilteredMRR,
    GeneralConv,
    GATv2Conv,
    GCN2Conv,
    GSAM,
    GatedGCNConv,
    GatedGraphConv,
    GENConv,
    GINEConv,
    GradualUnfreezing,
    GeneralizedCrossEntropyTask,
    Graph,
    GraphBatch,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    GraphConv,
    GraphTransformerEncoder,
    GraphTransformerEncoderLayer,
    GraphormerEncoder,
    GraphormerEncoderLayer,
    GraphSchema,
    GraphView,
    GPSLayer,
    GINConv,
    GMMConv,
    GPRGNNConv,
    HANConv,
    HEATConv,
    HGTConv,
    H2GCNConv,
    IdentityTemporalMessage,
    KarateClubDataset,
    LayerwiseLrDecay,
    LastMessageAggregator,
    LightGCNConv,
    Lookahead,
    LGConv,
    MixHopConv,
    LinkPredictionBatch,
    NodeBatch,
    Node2VecWalkSampler,
    LinkPredictionRecord,
    LinkPredictionTask,
    LinkNeighborSampler,
    NodeNeighborSampler,
    ListDataset,
    Loader,
    CandidateLinkSampler,
    HardNegativeLinkSampler,
    HitsAtK,
    LEConv,
    FullGraphSampler,
    FloodingTask,
    MFConv,
    NNConv,
    NodeSeedSubgraphSampler,
    PointNetConv,
    PointTransformerConv,
    Poly1CrossEntropyTask,
    Poly1EpsilonScheduler,
    PlanetoidDataset,
    RandomWalkSampler,
    ResGatedGraphConv,
    RDropTask,
    SampleRecord,
    ShaDowKHopSampler,
    SimpleConv,
    TemporalEventRecord,
    UniformNegativeLinkSampler,
    MessagePassing,
    NAGphormerEncoder,
    SSGConv,
    PDNConv,
    PNAConv,
    RGCNConv,
    RGATConv,
    SAM,
    MeanMessageAggregator,
    MRR,
    SAGEConv,
    SGFormerEncoder,
    SGFormerEncoderLayer,
    StochasticWeightAveraging,
    SuperGATConv,
    SymmetricCrossEntropyBetaScheduler,
    SymmetricCrossEntropyTask,
    Task,
    Metric,
    SplineConv,
    SGConv,
    TAGConv,
    TransformerConv,
    Trainer,
    TWIRLSConv,
    DirGNNConv,
    DeferredReweighting,
    EarlyStopping,
    ExponentialMovingAverage,
    GradientCentralization,
    LabelSmoothingScheduler,
    LdamMarginScheduler,
    LogitAdjustTauScheduler,
    ModelCheckpoint,
    PosWeightScheduler,
    WeightDecayScheduler,
    NodeClassificationTask,
    GraphClassificationTask,
    GroupRevRes,
    HistoryLogger,
    JSONLinesLogger,
    Logger,
    TemporalEventPredictionTask,
    TemporalEventBatch,
    TemporalNeighborSampler,
    TensorBoardLogger,
    TGNMemory,
    TGATEncoder,
    TGATLayer,
    TUDataset,
    TimeEncoder,
    TrainingHistory,
    WarmupCosineScheduler,
    WLConvContinuous,
    global_mean_pool,
    global_sum_pool,
    global_max_pool,
    __version__,
)
from scripts.contracts import RELEASE_VERSION


def test_package_exposes_broad_vgl_root_surface():
    assert AGNNConv.__name__ == "AGNNConv"
    assert ASAM.__name__ == "ASAM"
    assert Accuracy.__name__ == "Accuracy"
    assert AdaptiveGradientClipping.__name__ == "AdaptiveGradientClipping"
    assert AntiSymmetricConv.__name__ == "AntiSymmetricConv"
    assert APPNPConv.__name__ == "APPNPConv"
    assert ARMAConv.__name__ == "ARMAConv"
    assert BernConv.__name__ == "BernConv"
    assert BootstrapBetaScheduler.__name__ == "BootstrapBetaScheduler"
    assert BootstrapTask.__name__ == "BootstrapTask"
    assert CSVLogger.__name__ == "CSVLogger"
    assert CGConv.__name__ == "CGConv"
    assert ChebConv.__name__ == "ChebConv"
    assert ClusterData.__name__ == "ClusterData"
    assert ClusterGCNConv.__name__ == "ClusterGCNConv"
    assert ClusterLoader.__name__ == "ClusterLoader"
    assert ConsoleLogger.__name__ == "ConsoleLogger"
    assert ConfidencePenaltyScheduler.__name__ == "ConfidencePenaltyScheduler"
    assert ConfidencePenaltyTask.__name__ == "ConfidencePenaltyTask"
    assert DatasetRegistry.__name__ == "DatasetRegistry"
    assert DAGNNConv.__name__ == "DAGNNConv"
    assert DNAConv.__name__ == "DNAConv"
    assert EdgeConv.__name__ == "EdgeConv"
    assert EGConv.__name__ == "EGConv"
    assert FAConv.__name__ == "FAConv"
    assert FAGCNConv.__name__ == "FAGCNConv"
    assert FiLMConv.__name__ == "FiLMConv"
    assert FeaStConv.__name__ == "FeaStConv"
    assert FocalGammaScheduler.__name__ == "FocalGammaScheduler"
    assert FloodingLevelScheduler.__name__ == "FloodingLevelScheduler"
    assert GeneralizedCrossEntropyScheduler.__name__ == "GeneralizedCrossEntropyScheduler"
    assert GradientAccumulationScheduler.__name__ == "GradientAccumulationScheduler"
    assert FilteredHitsAtK.__name__ == "FilteredHitsAtK"
    assert FilteredMRR.__name__ == "FilteredMRR"
    assert SymmetricCrossEntropyBetaScheduler.__name__ == "SymmetricCrossEntropyBetaScheduler"
    assert FloodingTask.__name__ == "FloodingTask"
    assert GradientNoiseInjection.__name__ == "GradientNoiseInjection"
    assert GradientValueClipping.__name__ == "GradientValueClipping"
    assert GeneralizedCrossEntropyTask.__name__ == "GeneralizedCrossEntropyTask"
    assert GATConv.__name__ == "GATConv"
    assert GeneralConv.__name__ == "GeneralConv"
    assert GATv2Conv.__name__ == "GATv2Conv"
    assert GCN2Conv.__name__ == "GCN2Conv"
    assert GSAM.__name__ == "GSAM"
    assert GatedGCNConv.__name__ == "GatedGCNConv"
    assert GatedGraphConv.__name__ == "GatedGraphConv"
    assert GENConv.__name__ == "GENConv"
    assert GINEConv.__name__ == "GINEConv"
    assert Graph.__name__ == "Graph"
    assert GraphBatch.__name__ == "GraphBatch"
    assert GraphSAINTEdgeSampler.__name__ == "GraphSAINTEdgeSampler"
    assert GraphSAINTNodeSampler.__name__ == "GraphSAINTNodeSampler"
    assert GraphSAINTRandomWalkSampler.__name__ == "GraphSAINTRandomWalkSampler"
    assert GraphConv.__name__ == "GraphConv"
    assert GraphTransformerEncoder.__name__ == "GraphTransformerEncoder"
    assert GraphTransformerEncoderLayer.__name__ == "GraphTransformerEncoderLayer"
    assert GraphormerEncoder.__name__ == "GraphormerEncoder"
    assert GraphormerEncoderLayer.__name__ == "GraphormerEncoderLayer"
    assert GMMConv.__name__ == "GMMConv"
    assert GPRGNNConv.__name__ == "GPRGNNConv"
    assert GPSLayer.__name__ == "GPSLayer"
    assert HANConv.__name__ == "HANConv"
    assert HEATConv.__name__ == "HEATConv"
    assert LinkPredictionBatch.__name__ == "LinkPredictionBatch"
    assert NodeBatch.__name__ == "NodeBatch"
    assert HardNegativeLinkSampler.__name__ == "HardNegativeLinkSampler"
    assert HitsAtK.__name__ == "HitsAtK"
    assert LEConv.__name__ == "LEConv"
    assert LightGCNConv.__name__ == "LightGCNConv"
    assert LinkNeighborSampler.__name__ == "LinkNeighborSampler"
    assert NodeNeighborSampler.__name__ == "NodeNeighborSampler"
    assert MFConv.__name__ == "MFConv"
    assert MixHopConv.__name__ == "MixHopConv"
    assert MRR.__name__ == "MRR"
    assert NNConv.__name__ == "NNConv"
    assert TemporalEventBatch.__name__ == "TemporalEventBatch"
    assert TemporalNeighborSampler.__name__ == "TemporalNeighborSampler"
    assert TensorBoardLogger.__name__ == "TensorBoardLogger"
    assert TUDataset.__name__ == "TUDataset"
    assert UniformNegativeLinkSampler.__name__ == "UniformNegativeLinkSampler"
    assert PointNetConv.__name__ == "PointNetConv"
    assert PointTransformerConv.__name__ == "PointTransformerConv"
    assert Poly1CrossEntropyTask.__name__ == "Poly1CrossEntropyTask"
    assert Poly1EpsilonScheduler.__name__ == "Poly1EpsilonScheduler"
    assert GINConv.__name__ == "GINConv"
    assert HGTConv.__name__ == "HGTConv"
    assert GraphSchema.__name__ == "GraphSchema"
    assert GraphView.__name__ == "GraphView"
    assert H2GCNConv.__name__ == "H2GCNConv"
    assert IdentityTemporalMessage.__name__ == "IdentityTemporalMessage"
    assert KarateClubDataset.__name__ == "KarateClubDataset"
    assert LastMessageAggregator.__name__ == "LastMessageAggregator"
    assert ListDataset.__name__ == "ListDataset"
    assert LGConv.__name__ == "LGConv"
    assert Loader.__name__ == "Loader"
    assert Lookahead.__name__ == "Lookahead"
    assert CandidateLinkSampler.__name__ == "CandidateLinkSampler"
    assert FullGraphSampler.__name__ == "FullGraphSampler"
    assert MeanMessageAggregator.__name__ == "MeanMessageAggregator"
    assert NodeSeedSubgraphSampler.__name__ == "NodeSeedSubgraphSampler"
    assert Node2VecWalkSampler.__name__ == "Node2VecWalkSampler"
    assert PlanetoidDataset.__name__ == "PlanetoidDataset"
    assert RandomWalkSampler.__name__ == "RandomWalkSampler"
    assert ResGatedGraphConv.__name__ == "ResGatedGraphConv"
    assert RDropTask.__name__ == "RDropTask"
    assert LinkPredictionRecord.__name__ == "LinkPredictionRecord"
    assert SampleRecord.__name__ == "SampleRecord"
    assert ShaDowKHopSampler.__name__ == "ShaDowKHopSampler"
    assert SimpleConv.__name__ == "SimpleConv"
    assert TemporalEventRecord.__name__ == "TemporalEventRecord"
    assert TGNMemory.__name__ == "TGNMemory"
    assert TimeEncoder.__name__ == "TimeEncoder"
    assert TGATLayer.__name__ == "TGATLayer"
    assert TGATEncoder.__name__ == "TGATEncoder"
    assert MessagePassing.__name__ == "MessagePassing"
    assert NAGphormerEncoder.__name__ == "NAGphormerEncoder"
    assert PDNConv.__name__ == "PDNConv"
    assert PNAConv.__name__ == "PNAConv"
    assert RGCNConv.__name__ == "RGCNConv"
    assert RGATConv.__name__ == "RGATConv"
    assert SAM.__name__ == "SAM"
    assert SAGEConv.__name__ == "SAGEConv"
    assert SGFormerEncoder.__name__ == "SGFormerEncoder"
    assert SGFormerEncoderLayer.__name__ == "SGFormerEncoderLayer"
    assert SSGConv.__name__ == "SSGConv"
    assert SuperGATConv.__name__ == "SuperGATConv"
    assert SGConv.__name__ == "SGConv"
    assert SplineConv.__name__ == "SplineConv"
    assert TAGConv.__name__ == "TAGConv"
    assert Task.__name__ == "Task"
    assert Metric.__name__ == "Metric"
    assert TransformerConv.__name__ == "TransformerConv"
    assert Trainer.__name__ == "Trainer"
    assert TWIRLSConv.__name__ == "TWIRLSConv"
    assert DirGNNConv.__name__ == "DirGNNConv"
    assert DeferredReweighting.__name__ == "DeferredReweighting"
    assert EarlyStopping.__name__ == "EarlyStopping"
    assert ExponentialMovingAverage.__name__ == "ExponentialMovingAverage"
    assert GradientCentralization.__name__ == "GradientCentralization"
    assert LabelSmoothingScheduler.__name__ == "LabelSmoothingScheduler"
    assert LdamMarginScheduler.__name__ == "LdamMarginScheduler"
    assert LogitAdjustTauScheduler.__name__ == "LogitAdjustTauScheduler"
    assert ModelCheckpoint.__name__ == "ModelCheckpoint"
    assert PosWeightScheduler.__name__ == "PosWeightScheduler"
    assert WeightDecayScheduler.__name__ == "WeightDecayScheduler"
    assert GradualUnfreezing.__name__ == "GradualUnfreezing"
    assert HistoryLogger.__name__ == "HistoryLogger"
    assert JSONLinesLogger.__name__ == "JSONLinesLogger"
    assert LayerwiseLrDecay.__name__ == "LayerwiseLrDecay"
    assert Logger.__name__ == "Logger"
    assert StochasticWeightAveraging.__name__ == "StochasticWeightAveraging"
    assert SymmetricCrossEntropyTask.__name__ == "SymmetricCrossEntropyTask"
    assert TrainingHistory.__name__ == "TrainingHistory"
    assert WarmupCosineScheduler.__name__ == "WarmupCosineScheduler"
    assert WLConvContinuous.__name__ == "WLConvContinuous"
    assert NodeClassificationTask.__name__ == "NodeClassificationTask"
    assert GraphClassificationTask.__name__ == "GraphClassificationTask"
    assert GroupRevRes.__name__ == "GroupRevRes"
    assert LinkPredictionTask.__name__ == "LinkPredictionTask"
    assert FloodingTask.__name__ == "FloodingTask"
    assert TemporalEventPredictionTask.__name__ == "TemporalEventPredictionTask"
    assert callable(global_mean_pool)
    assert callable(global_sum_pool)
    assert callable(global_max_pool)
    assert __version__ == RELEASE_VERSION
