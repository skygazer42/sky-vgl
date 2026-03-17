import pytest
import torch

from vgl.metrics import build_metric as domain_build_metric
from vgl.train.metrics import Accuracy, build_metric


def test_legacy_train_package_reexports_build_metric():
    from vgl.train import build_metric as legacy_build_metric

    assert legacy_build_metric is domain_build_metric


def test_accuracy_handles_multiclass_logits():
    metric = Accuracy()

    metric.update(
        torch.tensor([[0.1, 0.9], [0.8, 0.2]]),
        torch.tensor([1, 0]),
    )

    assert metric.compute() == 1.0


def test_accuracy_handles_binary_logits():
    metric = Accuracy()

    metric.update(
        torch.tensor([1.5, -0.5, 0.1]),
        torch.tensor([1, 0, 1]),
    )

    assert metric.compute() == 1.0


def test_accuracy_rejects_shape_mismatch():
    metric = Accuracy()

    with pytest.raises(ValueError, match="shape"):
        metric.update(torch.randn(2, 3), torch.tensor([1]))


def test_build_metric_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported metric"):
        build_metric("f1")
