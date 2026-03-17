from vgl import DataLoader, Graph, Loader
from vgl.core.graph import Graph as LegacyGraph
from vgl.data.loader import Loader as LegacyLoader


def test_domain_packages_expose_preferred_import_paths():
    from vgl.dataloading import DataLoader as DomainDataLoader
    from vgl.dataloading import FullGraphSampler, ListDataset
    from vgl.engine import (
        ASAM,
        AdaptiveGradientClipping,
        CHECKPOINT_FORMAT,
        CHECKPOINT_FORMAT_VERSION,
        Callback,
        DeferredReweighting,
        EarlyStopping,
        GradientCentralization,
        GSAM,
        GradualUnfreezing,
        HistoryLogger,
        LayerwiseLrDecay,
        SAM,
        StopTraining,
        TrainingHistory,
        Trainer,
        WarmupCosineScheduler,
        load_checkpoint,
        restore_checkpoint,
        save_checkpoint,
    )
    from vgl.graph import Graph as DomainGraph
    from vgl.graph import GraphBatch, GraphSchema, GraphView
    from vgl.graph import LinkPredictionBatch, TemporalEventBatch
    from vgl.metrics import Accuracy, Metric, build_metric
    from vgl.tasks import (
        GraphClassificationTask,
        LinkPredictionTask,
        NodeClassificationTask,
        RDropTask,
        Task,
        TemporalEventPredictionTask,
    )
    from vgl.transforms import IdentityTransform
    from vgl.train import ASAM as LegacyASAM
    from vgl.train import GSAM as LegacyGSAM
    from vgl.train import RDropTask as LegacyRDropTask
    from vgl.train import SAM as LegacySAM

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
    assert Callback.__name__ == "Callback"
    assert DeferredReweighting.__name__ == "DeferredReweighting"
    assert EarlyStopping.__name__ == "EarlyStopping"
    assert GradientCentralization.__name__ == "GradientCentralization"
    assert GradualUnfreezing.__name__ == "GradualUnfreezing"
    assert HistoryLogger.__name__ == "HistoryLogger"
    assert GSAM.__name__ == "GSAM"
    assert LayerwiseLrDecay.__name__ == "LayerwiseLrDecay"
    assert LegacyASAM is ASAM
    assert LegacyGSAM is GSAM
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
