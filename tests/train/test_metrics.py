import pytest
import torch

from vgl.metrics import build_metric as domain_build_metric
from vgl.train.metrics import Accuracy, FilteredHitsAtK, FilteredMRR, HitsAtK, MRR, build_metric


class _StagePrefixedMetric:
    name = "val_accuracy"

    def reset(self):
        return None

    def update(self, predictions, targets, **kwargs):
        del predictions, targets, kwargs
        return None

    def compute(self):
        return 1.0


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


def test_accuracy_avoids_tensor_item(monkeypatch):
    metric = Accuracy()

    def fail_item(self):
        raise AssertionError("Accuracy should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    metric.update(
        torch.tensor([1.5, -0.5, 0.1]),
        torch.tensor([1, 0, 1]),
    )

    assert metric.compute() == 1.0


def test_accuracy_avoids_tensor_int(monkeypatch):
    metric = Accuracy()

    def fail_int(self):
        raise AssertionError("Accuracy should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    metric.update(
        torch.tensor([1.5, -0.5, 0.1]),
        torch.tensor([1, 0, 1]),
    )

    assert metric.compute() == 1.0


def test_accuracy_rejects_shape_mismatch():
    metric = Accuracy()

    with pytest.raises(ValueError, match="shape"):
        metric.update(torch.randn(2, 3), torch.tensor([1]))


def test_mrr_computes_mean_reciprocal_rank_from_query_groups():
    class Batch:
        query_index = torch.tensor([0, 0, 0, 1, 1, 1])

    metric = MRR()
    metric.update(
        torch.tensor([3.0, 1.0, 0.0, 0.1, 0.7, 0.2]),
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx((1.0 + (1.0 / 3.0)) / 2.0)


def test_hits_at_k_computes_hit_rate_from_query_groups():
    class Batch:
        query_index = torch.tensor([0, 0, 0, 1, 1, 1])

    hits1 = HitsAtK(1)
    hits3 = HitsAtK(3)
    predictions = torch.tensor([3.0, 1.0, 0.0, 0.1, 0.7, 0.2])
    targets = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])

    hits1.update(predictions, targets, batch=Batch())
    hits3.update(predictions, targets, batch=Batch())

    assert hits1.compute() == pytest.approx(0.5)
    assert hits3.compute() == pytest.approx(1.0)


def test_hits_at_k_accepts_tensor_k_without_tensor_int(monkeypatch):
    class Batch:
        query_index = torch.tensor([0, 0, 0, 1, 1, 1])

    def fail_int(self):
        raise AssertionError("HitsAtK k should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    metric = HitsAtK(torch.tensor(3))
    metric.update(
        torch.tensor([3.0, 1.0, 0.0, 0.1, 0.7, 0.2]),
        torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx(1.0)


def test_ranking_metrics_require_query_groups():
    metric = MRR()

    with pytest.raises(ValueError, match="query_index"):
        metric.update(torch.tensor([1.0, 0.0]), torch.tensor([1.0, 0.0]))


def test_filtered_mrr_ignores_masked_candidates():
    class Batch:
        query_index = torch.tensor([0, 0, 0])
        filter_mask = torch.tensor([False, True, False])

    metric = FilteredMRR()
    metric.update(
        torch.tensor([0.8, 0.9, 0.1]),
        torch.tensor([1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx(1.0)


def test_filtered_hits_at_k_ignores_masked_candidates():
    class Batch:
        query_index = torch.tensor([0, 0, 0])
        filter_mask = torch.tensor([False, True, False])

    metric = FilteredHitsAtK(1)
    metric.update(
        torch.tensor([0.8, 0.9, 0.1]),
        torch.tensor([1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx(1.0)


def test_filtered_hits_at_k_accepts_tensor_k_without_tensor_int(monkeypatch):
    class Batch:
        query_index = torch.tensor([0, 0, 0])
        filter_mask = torch.tensor([False, True, False])

    def fail_int(self):
        raise AssertionError("FilteredHitsAtK k should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    metric = FilteredHitsAtK(torch.tensor(1))
    metric.update(
        torch.tensor([0.8, 0.9, 0.1]),
        torch.tensor([1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx(1.0)


def test_mrr_handles_non_consecutive_query_indices():
    class Batch:
        query_index = torch.tensor([0, 1, 1, 0])

    metric = MRR()
    metric.update(
        torch.tensor([0.9, 0.2, 0.8, 0.1]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx((1.0 + 0.5) / 2.0)


def test_mrr_avoids_tensor_tolist_for_query_ids(monkeypatch):
    class Batch:
        query_index = torch.tensor([0, 1, 1, 0])

    def fail_tolist(self):
        raise AssertionError("Ranking metrics should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    metric = MRR()
    metric.update(
        torch.tensor([0.9, 0.2, 0.8, 0.1]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx((1.0 + 0.5) / 2.0)


def test_mrr_avoids_tensor_item(monkeypatch):
    class Batch:
        query_index = torch.tensor([0, 1, 1, 0])

    def fail_item(self):
        raise AssertionError("Ranking metrics should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    metric = MRR()
    metric.update(
        torch.tensor([0.9, 0.2, 0.8, 0.1]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx((1.0 + 0.5) / 2.0)


def test_mrr_avoids_tensor_int(monkeypatch):
    class Batch:
        query_index = torch.tensor([0, 1, 1, 0])

    def fail_int(self):
        raise AssertionError("Ranking metrics should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    metric = MRR()
    metric.update(
        torch.tensor([0.9, 0.2, 0.8, 0.1]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx((1.0 + 0.5) / 2.0)


def test_mrr_avoids_torch_unique(monkeypatch):
    class Batch:
        query_index = torch.tensor([0, 1, 1, 0])

    def fail_unique(*args, **kwargs):
        raise AssertionError("Ranking metrics should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    metric = MRR()
    metric.update(
        torch.tensor([0.9, 0.2, 0.8, 0.1]),
        torch.tensor([1.0, 1.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx((1.0 + 0.5) / 2.0)


def test_filtered_mrr_handles_non_consecutive_query_indices():
    class Batch:
        query_index = torch.tensor([0, 1, 1, 0, 0])
        filter_mask = torch.tensor([False, False, False, False, True])

    metric = FilteredMRR()
    metric.update(
        torch.tensor([0.9, 0.2, 0.4, 0.1, 0.95]),
        torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
        batch=Batch(),
    )

    assert metric.compute() == pytest.approx((1.0 + 0.5) / 2.0)


def test_build_metric_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported metric"):
        build_metric("f1")


def test_build_metric_supports_ranking_metrics():
    assert build_metric("mrr").name == "mrr"
    assert build_metric("hits@10").name == "hits@10"
    assert build_metric("filtered_mrr").name == "filtered_mrr"
    assert build_metric("filtered_hits@10").name == "filtered_hits@10"


def test_build_metric_rejects_stage_prefixed_metric_names():
    with pytest.raises(ValueError, match="reserved stage prefixes"):
        build_metric(_StagePrefixedMetric())
