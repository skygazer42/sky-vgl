from vgl.engine.monitoring import validate_metric_name
from vgl.metrics.base import Metric as Metric
from vgl.metrics.base import MetricProtocol, MetricSpec
from vgl.metrics.classification import Accuracy as Accuracy
from vgl.metrics.classification import build_metric as _build_metric
from vgl.metrics.ranking import FilteredHitsAtK as FilteredHitsAtK
from vgl.metrics.ranking import FilteredMRR as FilteredMRR
from vgl.metrics.ranking import HitsAtK as HitsAtK
from vgl.metrics.ranking import MRR as MRR


def build_metric(metric: MetricSpec) -> MetricProtocol:
    built = _build_metric(metric)
    validate_metric_name(built.name)
    return built

__all__ = ["Accuracy", "FilteredHitsAtK", "FilteredMRR", "HitsAtK", "MRR", "Metric", "build_metric"]
