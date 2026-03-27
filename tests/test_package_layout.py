from vgl import Block, DataLoader, Graph, Loader
from vgl.core.graph import Graph as LegacyGraph
from vgl.data.loader import Loader as LegacyLoader


def test_domain_packages_expose_preferred_import_paths():
    from vgl.dataloading import DataLoader as DomainDataLoader
    from vgl.dataloading import (
        CandidateLinkSampler,
        FullGraphSampler,
        HardNegativeLinkSampler,
        LinkNeighborSampler,
        NodeNeighborSampler,
        ListDataset,
        TemporalNeighborSampler,
        UniformNegativeLinkSampler,
    )
    from vgl.engine import (
        ASAM,
        AdaptiveGradientClipping,
        BootstrapBetaScheduler,
        CSVLogger,
        CHECKPOINT_FORMAT,
        CHECKPOINT_FORMAT_VERSION,
        ConsoleLogger,
        Callback,
        ConfidencePenaltyScheduler,
        DeferredReweighting,
        EarlyStopping,
        FocalGammaScheduler,
        FloodingLevelScheduler,
        GeneralizedCrossEntropyScheduler,
        GradientAccumulationScheduler,
        GradientNoiseInjection,
        GradientValueClipping,
        GradientCentralization,
        GSAM,
        GradualUnfreezing,
        HistoryLogger,
        JSONLinesLogger,
        LabelSmoothingScheduler,
        LdamMarginScheduler,
        Logger,
        LogitAdjustTauScheduler,
        ModelCheckpoint,
        Poly1EpsilonScheduler,
        PosWeightScheduler,
        LayerwiseLrDecay,
        SAM,
        StopTraining,
        SymmetricCrossEntropyBetaScheduler,
        TensorBoardLogger,
        TrainingHistory,
        Trainer,
        WarmupCosineScheduler,
        WeightDecayScheduler,
        load_checkpoint,
        restore_checkpoint,
        save_checkpoint,
    )
    from vgl.graph import Block as DomainBlock
    from vgl.graph import Graph as DomainGraph
    from vgl.graph import GraphBatch, GraphSchema, GraphView, NodeBatch
    from vgl.graph import LinkPredictionBatch, TemporalEventBatch
    from vgl.metrics import Accuracy, FilteredHitsAtK, FilteredMRR, HitsAtK, Metric, MRR, build_metric
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
    from vgl.transforms import IdentityTransform, RandomLinkSplit
    from vgl.train import ASAM as LegacyASAM
    from vgl.train import BootstrapBetaScheduler as LegacyBootstrapBetaScheduler
    from vgl.train import BootstrapTask as LegacyBootstrapTask
    from vgl.train import CSVLogger as LegacyCSVLogger
    from vgl.train import ConsoleLogger as LegacyConsoleLogger
    from vgl.train import ConfidencePenaltyScheduler as LegacyConfidencePenaltyScheduler
    from vgl.train import ConfidencePenaltyTask as LegacyConfidencePenaltyTask
    from vgl.train import FilteredHitsAtK as LegacyFilteredHitsAtK
    from vgl.train import FilteredMRR as LegacyFilteredMRR
    from vgl.train import FloodingLevelScheduler as LegacyFloodingLevelScheduler
    from vgl.train import FloodingTask as LegacyFloodingTask
    from vgl.train import GeneralizedCrossEntropyScheduler as LegacyGeneralizedCrossEntropyScheduler
    from vgl.train import GradientAccumulationScheduler as LegacyGradientAccumulationScheduler
    from vgl.train import FocalGammaScheduler as LegacyFocalGammaScheduler
    from vgl.train import GeneralizedCrossEntropyTask as LegacyGeneralizedCrossEntropyTask
    from vgl.train import GSAM as LegacyGSAM
    from vgl.train import GradientNoiseInjection as LegacyGradientNoiseInjection
    from vgl.train import GradientValueClipping as LegacyGradientValueClipping
    from vgl.train import HitsAtK as LegacyHitsAtK
    from vgl.train import JSONLinesLogger as LegacyJSONLinesLogger
    from vgl.train import LdamMarginScheduler as LegacyLdamMarginScheduler
    from vgl.train import Logger as LegacyLogger
    from vgl.train import LogitAdjustTauScheduler as LegacyLogitAdjustTauScheduler
    from vgl.train import ModelCheckpoint as LegacyModelCheckpoint
    from vgl.train import MRR as LegacyMRR
    from vgl.train import Poly1CrossEntropyTask as LegacyPoly1CrossEntropyTask
    from vgl.train import Poly1EpsilonScheduler as LegacyPoly1EpsilonScheduler
    from vgl.train import PosWeightScheduler as LegacyPosWeightScheduler
    from vgl.train import RDropTask as LegacyRDropTask
    from vgl.train import SAM as LegacySAM
    from vgl.train import SymmetricCrossEntropyBetaScheduler as LegacySymmetricCrossEntropyBetaScheduler
    from vgl.train import SymmetricCrossEntropyTask as LegacySymmetricCrossEntropyTask
    from vgl.train import TensorBoardLogger as LegacyTensorBoardLogger
    from vgl.train import WeightDecayScheduler as LegacyWeightDecayScheduler
    from vgl.data import CandidateLinkSampler as LegacyCandidateLinkSampler
    from vgl.data import HardNegativeLinkSampler as LegacyHardNegativeLinkSampler
    from vgl.data import LinkNeighborSampler as LegacyLinkNeighborSampler
    from vgl.data import NodeNeighborSampler as LegacyNodeNeighborSampler
    from vgl.data import TemporalNeighborSampler as LegacyTemporalNeighborSampler
    from vgl.data import UniformNegativeLinkSampler as LegacyUniformNegativeLinkSampler

    assert DomainBlock is Block
    assert DomainGraph is Graph
    assert Graph is LegacyGraph
    assert DomainDataLoader is DataLoader
    assert DataLoader is Loader
    assert Loader is LegacyLoader
    assert GraphBatch.__name__ == "GraphBatch"
    assert GraphSchema.__name__ == "GraphSchema"
    assert GraphView.__name__ == "GraphView"
    assert LinkPredictionBatch.__name__ == "LinkPredictionBatch"
    assert NodeBatch.__name__ == "NodeBatch"
    assert TemporalEventBatch.__name__ == "TemporalEventBatch"
    assert CandidateLinkSampler.__name__ == "CandidateLinkSampler"
    assert FullGraphSampler.__name__ == "FullGraphSampler"
    assert HardNegativeLinkSampler.__name__ == "HardNegativeLinkSampler"
    assert LinkNeighborSampler.__name__ == "LinkNeighborSampler"
    assert NodeNeighborSampler.__name__ == "NodeNeighborSampler"
    assert TemporalNeighborSampler.__name__ == "TemporalNeighborSampler"
    assert UniformNegativeLinkSampler.__name__ == "UniformNegativeLinkSampler"
    assert ListDataset.__name__ == "ListDataset"
    assert ASAM.__name__ == "ASAM"
    assert AdaptiveGradientClipping.__name__ == "AdaptiveGradientClipping"
    assert BootstrapBetaScheduler.__name__ == "BootstrapBetaScheduler"
    assert BootstrapTask.__name__ == "BootstrapTask"
    assert CSVLogger.__name__ == "CSVLogger"
    assert Callback.__name__ == "Callback"
    assert Logger.__name__ == "Logger"
    assert ConsoleLogger.__name__ == "ConsoleLogger"
    assert ConfidencePenaltyScheduler.__name__ == "ConfidencePenaltyScheduler"
    assert DeferredReweighting.__name__ == "DeferredReweighting"
    assert EarlyStopping.__name__ == "EarlyStopping"
    assert FocalGammaScheduler.__name__ == "FocalGammaScheduler"
    assert FloodingLevelScheduler.__name__ == "FloodingLevelScheduler"
    assert GeneralizedCrossEntropyScheduler.__name__ == "GeneralizedCrossEntropyScheduler"
    assert SymmetricCrossEntropyBetaScheduler.__name__ == "SymmetricCrossEntropyBetaScheduler"
    assert GradientAccumulationScheduler.__name__ == "GradientAccumulationScheduler"
    assert GradientNoiseInjection.__name__ == "GradientNoiseInjection"
    assert GradientValueClipping.__name__ == "GradientValueClipping"
    assert GradientCentralization.__name__ == "GradientCentralization"
    assert GradualUnfreezing.__name__ == "GradualUnfreezing"
    assert HistoryLogger.__name__ == "HistoryLogger"
    assert JSONLinesLogger.__name__ == "JSONLinesLogger"
    assert TensorBoardLogger.__name__ == "TensorBoardLogger"
    assert LabelSmoothingScheduler.__name__ == "LabelSmoothingScheduler"
    assert LdamMarginScheduler.__name__ == "LdamMarginScheduler"
    assert LogitAdjustTauScheduler.__name__ == "LogitAdjustTauScheduler"
    assert ModelCheckpoint.__name__ == "ModelCheckpoint"
    assert PosWeightScheduler.__name__ == "PosWeightScheduler"
    assert WeightDecayScheduler.__name__ == "WeightDecayScheduler"
    assert GSAM.__name__ == "GSAM"
    assert LayerwiseLrDecay.__name__ == "LayerwiseLrDecay"
    assert LegacyASAM is ASAM
    assert LegacyBootstrapBetaScheduler is BootstrapBetaScheduler
    assert LegacyBootstrapTask is BootstrapTask
    assert LegacyCSVLogger is CSVLogger
    assert LegacyConsoleLogger is ConsoleLogger
    assert LegacyConfidencePenaltyScheduler is ConfidencePenaltyScheduler
    assert LegacyFilteredHitsAtK is FilteredHitsAtK
    assert LegacyFilteredMRR is FilteredMRR
    assert LegacyFloodingLevelScheduler is FloodingLevelScheduler
    assert LegacyGeneralizedCrossEntropyScheduler is GeneralizedCrossEntropyScheduler
    assert LegacyGradientAccumulationScheduler is GradientAccumulationScheduler
    assert LegacySymmetricCrossEntropyBetaScheduler is SymmetricCrossEntropyBetaScheduler
    assert LegacyFocalGammaScheduler is FocalGammaScheduler
    assert LegacyGSAM is GSAM
    assert LegacyGradientNoiseInjection is GradientNoiseInjection
    assert LegacyGradientValueClipping is GradientValueClipping
    assert LegacyHitsAtK is HitsAtK
    assert LegacyJSONLinesLogger is JSONLinesLogger
    assert LegacyTensorBoardLogger is TensorBoardLogger
    assert LegacyLdamMarginScheduler is LdamMarginScheduler
    assert LegacyLogger is Logger
    assert LegacyLogitAdjustTauScheduler is LogitAdjustTauScheduler
    assert LegacyModelCheckpoint is ModelCheckpoint
    assert LegacyMRR is MRR
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
    assert FilteredHitsAtK.__name__ == "FilteredHitsAtK"
    assert FilteredMRR.__name__ == "FilteredMRR"
    assert HitsAtK.__name__ == "HitsAtK"
    assert Metric.__name__ == "Metric"
    assert MRR.__name__ == "MRR"
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
    assert RandomLinkSplit.__name__ == "RandomLinkSplit"
    assert LegacyCandidateLinkSampler is CandidateLinkSampler
    assert LegacyHardNegativeLinkSampler is HardNegativeLinkSampler
    assert LegacyLinkNeighborSampler is LinkNeighborSampler
    assert LegacyNodeNeighborSampler is NodeNeighborSampler
    assert LegacyTemporalNeighborSampler is TemporalNeighborSampler
    assert LegacyUniformNegativeLinkSampler is UniformNegativeLinkSampler


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


def test_foundation_packages_expose_namespace_exports():
    import vgl.ops as ops_module
    from vgl.ops import GraphTransform, TransformPipeline
    from vgl.ops import add_self_loops, adj, adj_external, adj_tensors, all_edges, compact_nodes, create_formats_, edge_ids, edge_subgraph, find_edges, formats, has_edges_between, inc, in_degrees, in_edges, in_subgraph, khop_nodes, khop_subgraph, laplacian, line_graph, metapath_reachable_graph, node_subgraph, num_edges, num_nodes, number_of_edges, number_of_nodes, out_degrees, out_edges, out_subgraph, predecessors, remove_self_loops, reverse, successors, to_bidirected, to_block, to_hetero_block, to_simple
    from vgl.ops import __all__ as ops_all
    from vgl.sparse import SparseLayout, SparseTensor
    from vgl.sparse import __all__ as sparse_all
    from vgl.sparse import degree, edge_softmax, from_edge_index, from_torch_sparse, sddmm, select_cols, select_rows, spmm, sum as sparse_sum, to_coo, to_csc, to_csr, to_torch_sparse, transpose
    from vgl.storage import FeatureStore, GraphStore, InMemoryGraphStore, InMemoryTensorStore, MmapTensorStore, TensorSlice, TensorStore
    from vgl.storage import __all__ as storage_all

    assert ops_all == ["GraphTransform", "TransformPipeline", "add_self_loops", "remove_self_loops", "to_bidirected", "to_simple", "reverse", "line_graph", "metapath_reachable_graph", "random_walk", "metapath_random_walk", "find_edges", "edge_ids", "has_edges_between", "num_nodes", "number_of_nodes", "num_edges", "number_of_edges", "all_edges", "formats", "create_formats_", "adj", "laplacian", "adj_external", "adj_tensors", "inc", "in_degrees", "out_degrees", "in_edges", "out_edges", "predecessors", "successors", "node_subgraph", "edge_subgraph", "in_subgraph", "out_subgraph", "khop_nodes", "khop_subgraph", "compact_nodes", "to_block", "to_hetero_block"]
    assert sparse_all == ["SparseLayout", "SparseTensor", "from_edge_index", "from_torch_sparse", "to_coo", "to_csr", "to_csc", "to_torch_sparse", "degree", "select_rows", "select_cols", "transpose", "sum", "spmm", "sddmm", "edge_softmax"]
    assert storage_all == ["TensorSlice", "TensorStore", "InMemoryTensorStore", "MmapTensorStore", "FeatureStore", "GraphStore", "InMemoryGraphStore"]
    assert SparseLayout.__name__ == "SparseLayout"
    assert SparseTensor.__name__ == "SparseTensor"
    assert callable(from_edge_index)
    assert callable(from_torch_sparse)
    assert callable(to_coo)
    assert callable(to_csr)
    assert callable(to_csc)
    assert callable(to_torch_sparse)
    assert GraphTransform.__name__ == "GraphTransform"
    assert TransformPipeline.__name__ == "TransformPipeline"
    assert callable(add_self_loops)
    assert callable(remove_self_loops)
    assert callable(to_bidirected)
    assert callable(to_simple)
    assert callable(reverse)
    assert callable(line_graph)
    assert callable(find_edges)
    assert callable(edge_ids)
    assert callable(has_edges_between)
    assert callable(num_nodes)
    assert callable(number_of_nodes)
    assert callable(num_edges)
    assert callable(number_of_edges)
    assert callable(all_edges)
    assert callable(formats)
    assert callable(create_formats_)
    assert callable(adj)
    assert callable(laplacian)
    assert callable(adj_external)
    assert callable(adj_tensors)
    assert callable(inc)
    assert callable(in_degrees)
    assert callable(in_edges)
    assert callable(out_degrees)
    assert callable(out_edges)
    assert callable(predecessors)
    assert callable(successors)
    assert callable(metapath_reachable_graph)
    assert callable(getattr(ops_module, "random_walk", None))
    assert callable(getattr(ops_module, "metapath_random_walk", None))
    assert callable(node_subgraph)
    assert callable(edge_subgraph)
    assert callable(khop_nodes)
    assert callable(khop_subgraph)
    assert callable(in_subgraph)
    assert callable(out_subgraph)
    assert callable(compact_nodes)
    assert callable(to_block)
    assert callable(to_hetero_block)
    assert TensorSlice.__name__ == "TensorSlice"
    assert TensorStore.__name__ == "TensorStore"
    assert InMemoryTensorStore.__name__ == "InMemoryTensorStore"
    assert MmapTensorStore.__name__ == "MmapTensorStore"
    assert FeatureStore.__name__ == "FeatureStore"
    assert GraphStore.__name__ == "GraphStore"
    assert InMemoryGraphStore.__name__ == "InMemoryGraphStore"
    assert callable(degree)
    assert callable(select_rows)
    assert callable(select_cols)
    assert callable(transpose)
    assert callable(sparse_sum)
    assert callable(spmm)
    assert callable(sddmm)
    assert callable(edge_softmax)


def test_compat_package_exposes_networkx_helpers():
    import vgl.compat as compat_module
    from vgl.compat import from_networkx, to_networkx

    assert callable(from_networkx)
    assert callable(to_networkx)
    assert callable(getattr(compat_module, "from_networkx", None))
    assert callable(getattr(compat_module, "to_networkx", None))


def test_compat_package_exposes_edge_list_helpers():
    import vgl.compat as compat_module
    from vgl.compat import from_edge_list, to_edge_list

    assert callable(from_edge_list)
    assert callable(to_edge_list)
    assert callable(getattr(compat_module, "from_edge_list", None))
    assert callable(getattr(compat_module, "to_edge_list", None))


def test_compat_package_exposes_edge_list_csv_helpers():
    import vgl.compat as compat_module
    from vgl.compat import from_edge_list_csv, to_edge_list_csv

    assert callable(from_edge_list_csv)
    assert callable(to_edge_list_csv)
    assert callable(getattr(compat_module, "from_edge_list_csv", None))
    assert callable(getattr(compat_module, "to_edge_list_csv", None))


def test_compat_package_exposes_csv_table_helpers():
    import vgl.compat as compat_module
    from vgl.compat import from_csv_tables, to_csv_tables

    assert callable(from_csv_tables)
    assert callable(to_csv_tables)
    assert callable(getattr(compat_module, "from_csv_tables", None))
    assert callable(getattr(compat_module, "to_csv_tables", None))


def test_foundation_packages_expose_in_degrees_and_out_degrees_exports():
    import vgl.ops as ops_module
    from vgl.ops import in_degrees, out_degrees
    from vgl.ops import __all__ as ops_all

    assert "in_degrees" in ops_all
    assert "out_degrees" in ops_all
    assert callable(in_degrees)
    assert callable(out_degrees)
    assert callable(getattr(ops_module, "in_degrees", None))
    assert callable(getattr(ops_module, "out_degrees", None))


def test_foundation_packages_expose_cardinality_and_all_edges_exports():
    import vgl.ops as ops_module
    from vgl.ops import all_edges, num_edges, num_nodes, number_of_edges, number_of_nodes
    from vgl.ops import __all__ as ops_all

    assert "num_nodes" in ops_all
    assert "number_of_nodes" in ops_all
    assert "num_edges" in ops_all
    assert "number_of_edges" in ops_all
    assert "all_edges" in ops_all
    assert callable(num_nodes)
    assert callable(number_of_nodes)
    assert callable(num_edges)
    assert callable(number_of_edges)
    assert callable(all_edges)
    assert callable(getattr(ops_module, "num_nodes", None))
    assert callable(getattr(ops_module, "number_of_nodes", None))
    assert callable(getattr(ops_module, "num_edges", None))
    assert callable(getattr(ops_module, "number_of_edges", None))
    assert callable(getattr(ops_module, "all_edges", None))


def test_foundation_packages_expose_incidence_export():
    import vgl.ops as ops_module
    from vgl.ops import inc
    from vgl.ops import __all__ as ops_all

    assert "inc" in ops_all
    assert callable(inc)
    assert callable(getattr(ops_module, "inc", None))


def test_foundation_packages_expose_adjacency_tensor_export():
    import vgl.ops as ops_module
    from vgl.ops import adj_tensors
    from vgl.ops import __all__ as ops_all

    assert "adj_tensors" in ops_all
    assert callable(adj_tensors)
    assert callable(getattr(ops_module, "adj_tensors", None))


def test_foundation_packages_expose_adjacency_sparse_view_export():
    import vgl.ops as ops_module
    from vgl.ops import adj
    from vgl.ops import __all__ as ops_all

    assert "adj" in ops_all
    assert callable(adj)
    assert callable(getattr(ops_module, "adj", None))


def test_foundation_packages_expose_external_adjacency_export():
    import vgl.ops as ops_module
    from vgl.ops import adj_external
    from vgl.ops import __all__ as ops_all

    assert "adj_external" in ops_all
    assert callable(adj_external)
    assert callable(getattr(ops_module, "adj_external", None))


def test_foundation_packages_expose_graph_format_state_exports():
    import vgl.ops as ops_module
    from vgl.ops import create_formats_, formats
    from vgl.ops import __all__ as ops_all

    assert "formats" in ops_all
    assert "create_formats_" in ops_all
    assert callable(formats)
    assert callable(create_formats_)
    assert callable(getattr(ops_module, "formats", None))
    assert callable(getattr(ops_module, "create_formats_", None))
