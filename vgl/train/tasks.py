from vgl.tasks.bootstrap import BootstrapTask as BootstrapTask
from vgl.tasks.confidence_penalty import ConfidencePenaltyTask as ConfidencePenaltyTask
from vgl.tasks.flooding import FloodingTask as FloodingTask
from vgl.tasks.generalized_cross_entropy import (
    GeneralizedCrossEntropyTask as GeneralizedCrossEntropyTask,
)
from vgl.tasks.graph_classification import GraphClassificationTask as GraphClassificationTask
from vgl.tasks.link_prediction import LinkPredictionTask as LinkPredictionTask
from vgl.tasks.node_classification import NodeClassificationTask as NodeClassificationTask
from vgl.tasks.poly1_cross_entropy import Poly1CrossEntropyTask as Poly1CrossEntropyTask
from vgl.tasks.rdrop import RDropTask as RDropTask
from vgl.tasks.symmetric_cross_entropy import (
    SymmetricCrossEntropyTask as SymmetricCrossEntropyTask,
)
from vgl.tasks.temporal_event_prediction import (
    TemporalEventPredictionTask as TemporalEventPredictionTask,
)

__all__ = [
    "BootstrapTask",
    "ConfidencePenaltyTask",
    "FloodingTask",
    "GeneralizedCrossEntropyTask",
    "NodeClassificationTask",
    "GraphClassificationTask",
    "LinkPredictionTask",
    "Poly1CrossEntropyTask",
    "RDropTask",
    "SymmetricCrossEntropyTask",
    "TemporalEventPredictionTask",
]
