from vgl.tasks.graph_classification import GraphClassificationTask as GraphClassificationTask
from vgl.tasks.link_prediction import LinkPredictionTask as LinkPredictionTask
from vgl.tasks.node_classification import NodeClassificationTask as NodeClassificationTask
from vgl.tasks.rdrop import RDropTask as RDropTask
from vgl.tasks.temporal_event_prediction import (
    TemporalEventPredictionTask as TemporalEventPredictionTask,
)

__all__ = [
    "NodeClassificationTask",
    "GraphClassificationTask",
    "LinkPredictionTask",
    "RDropTask",
    "TemporalEventPredictionTask",
]
