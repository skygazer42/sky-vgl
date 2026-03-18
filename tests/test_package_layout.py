from vgl import DataLoader, Graph, Loader
from vgl.core.graph import Graph as LegacyGraph
from vgl.data.loader import Loader as LegacyLoader


def test_domain_packages_expose_preferred_import_paths():
    from vgl.dataloading import DataLoader as DomainDataLoader
    from vgl.dataloading import FullGraphSampler, ListDataset
    from vgl.engine import (
        ASAM,
        AdaptiveGradientClipping,
        BootstrapBetaScheduler,
        CHECKPOINT_FORMAT,
        CHECKPOINT_FORMAT_VERSION,
        Callback,
        ConfidencePenaltyScheduler,
        DeferredReweighting,
        EarlyStopping,
        FocalGammaScheduler,
        FloodingLevelScheduler,
        GeneralizedCrossEntropyScheduler,
        GradientNoiseInjection,
        GradientValueClipping,
        GradientCentralization,
        GSAM,
        GradualUnfreezing,
        HistoryLogger,
        LabelSmoothingScheduler,
        LdamMarginScheduler,
        LogitAdjustTauScheduler,
        Poly1EpsilonScheduler,
        PosWeightScheduler,
        LayerwiseLrDecay,
        SAM,
        StopTraining,
        SymmetricCrossEntropyBetaScheduler,
        TrainingHistory,
        Trainer,
        WarmupCosineScheduler,
        WeightDecayScheduler,
        load_checkpoint,
        restore_checkpoint,
        save_checkpoint,
    )
    from vgl.graph import Graph as DomainGraph
    from vgl.graph import GraphBatch, GraphSchema, GraphView
    from vgl.graph import LinkPredictionBatch, TemporalEventBatch
    from vgl.metrics import Accuracy, Metric, build_metric
    from vgl.tasks import (
        BootstrapTask,
        ConfidencePenaltyTask,
        FloodingTask,
        GeneralizedCrossEntropyTask,
        GraphClassificationTask,
        LinkPredictionTask,
        NodeClassificationTask,
        Poly1CrossEntropyTask,
        RDropTask,
        SymmetricCrossEntropyTask,
        Task,
        TemporalEventPredictionTask,
    )
    from vgl.transforms import IdentityTransform
    from vgl.train import ASAM as LegacyASAM
    from vgl.train import BootstrapBetaScheduler as LegacyBootstrapBetaScheduler
    from vgl.train import BootstrapTask as LegacyBootstrapTask
    from vgl.train import ConfidencePenaltyScheduler as LegacyConfidencePenaltyScheduler
    from vgl.train import ConfidencePenaltyTask as LegacyConfidencePenaltyTask
    from vgl.train import FloodingLevelScheduler as LegacyFloodingLevelScheduler
    from vgl.train import FloodingTask as LegacyFloodingTask
    from vgl.train import GeneralizedCrossEntropyScheduler as LegacyGeneralizedCrossEntropyScheduler
    from vgl.train import FocalGammaScheduler as LegacyFocalGammaScheduler
    from vgl.train import GeneralizedCrossEntropyTask as LegacyGeneralizedCrossEntropyTask
    from vgl.train import GSAM as LegacyGSAM
    from vgl.train import GradientNoiseInjection as LegacyGradientNoiseInjection
    from vgl.train import GradientValueClipping as LegacyGradientValueClipping
    from vgl.train import LdamMarginScheduler as LegacyLdamMarginScheduler
    from vgl.train import LogitAdjustTauScheduler as LegacyLogitAdjustTauScheduler
    from vgl.train import Poly1CrossEntropyTask as LegacyPoly1CrossEntropyTask
    from vgl.train import Poly1EpsilonScheduler as LegacyPoly1EpsilonScheduler
    from vgl.train import PosWeightScheduler as LegacyPosWeightScheduler
    from vgl.train import RDropTask as LegacyRDropTask
    from vgl.train import SAM as LegacySAM
    from vgl.train import SymmetricCrossEntropyBetaScheduler as LegacySymmetricCrossEntropyBetaScheduler
    from vgl.train import SymmetricCrossEntropyTask as LegacySymmetricCrossEntropyTask
    from vgl.train import WeightDecayScheduler as LegacyWeightDecayScheduler

    assert DomainGraph is Graph
    assert Graph is LegacyGraph
    assert DomainDataLoader is DataLoader
    assert DataLoader is Loader
    assert Loader is LegacyLoader
    assert GraphBatch.__name__ == "GraphBatch"
    assert GraphSchema.__name__ == "GraphSchema"
    assert GraphView.__name__ == "GraphView"
    assert LinkPredictionBatch.__name__ == "LinkPredictionBatch"
    assert TemporalEventBatch.__name__ == "TemporalEventBatch"
    assert FullGraphSampler.__name__ == "FullGraphSampler"
    assert ListDataset.__name__ == "ListDataset"
    assert ASAM.__name__ == "ASAM"
    assert AdaptiveGradientClipping.__name__ == "AdaptiveGradientClipping"
    assert BootstrapBetaScheduler.__name__ == "BootstrapBetaScheduler"
    assert BootstrapTask.__name__ == "BootstrapTask"
    assert Callback.__name__ == "Callback"
    assert ConfidencePenaltyScheduler.__name__ == "ConfidencePenaltyScheduler"
    assert DeferredReweighting.__name__ == "DeferredReweighting"
    assert EarlyStopping.__name__ == "EarlyStopping"
    assert FocalGammaScheduler.__name__ == "FocalGammaScheduler"
    assert FloodingLevelScheduler.__name__ == "FloodingLevelScheduler"
    assert GeneralizedCrossEntropyScheduler.__name__ == "GeneralizedCrossEntropyScheduler"
    assert SymmetricCrossEntropyBetaScheduler.__name__ == "SymmetricCrossEntropyBetaScheduler"
    assert GradientNoiseInjection.__name__ == "GradientNoiseInjection"
    assert GradientValueClipping.__name__ == "GradientValueClipping"
    assert GradientCentralization.__name__ == "GradientCentralization"
    assert GradualUnfreezing.__name__ == "GradualUnfreezing"
    assert HistoryLogger.__name__ == "HistoryLogger"
    assert LabelSmoothingScheduler.__name__ == "LabelSmoothingScheduler"
    assert LdamMarginScheduler.__name__ == "LdamMarginScheduler"
    assert LogitAdjustTauScheduler.__name__ == "LogitAdjustTauScheduler"
    assert PosWeightScheduler.__name__ == "PosWeightScheduler"
    assert WeightDecayScheduler.__name__ == "WeightDecayScheduler"
    assert GSAM.__name__ == "GSAM"
    assert LayerwiseLrDecay.__name__ == "LayerwiseLrDecay"
    assert LegacyASAM is ASAM
    assert LegacyBootstrapBetaScheduler is BootstrapBetaScheduler
    assert LegacyBootstrapTask is BootstrapTask
    assert LegacyConfidencePenaltyScheduler is ConfidencePenaltyScheduler
    assert LegacyFloodingLevelScheduler is FloodingLevelScheduler
    assert LegacyGeneralizedCrossEntropyScheduler is GeneralizedCrossEntropyScheduler
    assert LegacySymmetricCrossEntropyBetaScheduler is SymmetricCrossEntropyBetaScheduler
    assert LegacyFocalGammaScheduler is FocalGammaScheduler
    assert LegacyGSAM is GSAM
    assert LegacyGradientNoiseInjection is GradientNoiseInjection
    assert LegacyGradientValueClipping is GradientValueClipping
    assert LegacyLdamMarginScheduler is LdamMarginScheduler
    assert LegacyLogitAdjustTauScheduler is LogitAdjustTauScheduler
    assert LegacyPosWeightScheduler is PosWeightScheduler
    assert LegacyWeightDecayScheduler is WeightDecayScheduler
    assert SAM.__name__ == "SAM"
    assert LegacySAM is SAM
    assert StopTraining.__name__ == "StopTraining"
    assert TrainingHistory.__name__ == "TrainingHistory"
    assert WarmupCosineScheduler.__name__ == "WarmupCosineScheduler"
    assert CHECKPOINT_FORMAT == "vgl.trainer_checkpoint"
    assert CHECKPOINT_FORMAT_VERSION == 1
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)
    assert callable(restore_checkpoint)
    assert Trainer.__name__ == "Trainer"
    assert Accuracy.__name__ == "Accuracy"
    assert Metric.__name__ == "Metric"
    assert callable(build_metric)
    assert RDropTask.__name__ == "RDropTask"
    assert LegacyRDropTask is RDropTask
    assert ConfidencePenaltyTask.__name__ == "ConfidencePenaltyTask"
    assert LegacyConfidencePenaltyTask is ConfidencePenaltyTask
    assert FloodingTask.__name__ == "FloodingTask"
    assert LegacyFloodingTask is FloodingTask
    assert GeneralizedCrossEntropyTask.__name__ == "GeneralizedCrossEntropyTask"
    assert LegacyGeneralizedCrossEntropyTask is GeneralizedCrossEntropyTask
    assert Poly1CrossEntropyTask.__name__ == "Poly1CrossEntropyTask"
    assert Poly1EpsilonScheduler.__name__ == "Poly1EpsilonScheduler"
    assert LegacyPoly1CrossEntropyTask is Poly1CrossEntropyTask
    assert LegacyPoly1EpsilonScheduler is Poly1EpsilonScheduler
    assert SymmetricCrossEntropyTask.__name__ == "SymmetricCrossEntropyTask"
    assert LegacySymmetricCrossEntropyTask is SymmetricCrossEntropyTask
    assert Task.__name__ == "Task"
    assert NodeClassificationTask.__name__ == "NodeClassificationTask"
    assert GraphClassificationTask.__name__ == "GraphClassificationTask"
    assert LinkPredictionTask.__name__ == "LinkPredictionTask"
    assert TemporalEventPredictionTask.__name__ == "TemporalEventPredictionTask"
    assert IdentityTransform.__name__ == "IdentityTransform"


def test_legacy_train_package_reexports_engine_evaluator():
    from vgl.engine import Evaluator
    from vgl.train import Evaluator as LegacyEvaluator

    assert LegacyEvaluator is Evaluator


def test_legacy_data_package_reexports_sampler_base():
    from vgl.data import Sampler as LegacySampler
    from vgl.dataloading import Sampler

    assert LegacySampler is Sampler


def test_legacy_core_package_reexports_graph_support_types():
    from vgl.core import EdgeStore as LegacyEdgeStore
    from vgl.core import GNNError as LegacyGNNError
    from vgl.core import NodeStore as LegacyNodeStore
    from vgl.core import SchemaError as LegacySchemaError
    from vgl.graph import EdgeStore, GNNError, NodeStore, SchemaError

    assert LegacyNodeStore is NodeStore
    assert LegacyEdgeStore is EdgeStore
    assert LegacyGNNError is GNNError
    assert LegacySchemaError is SchemaError
