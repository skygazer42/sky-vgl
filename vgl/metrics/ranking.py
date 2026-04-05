from vgl.metrics.base import Metric


def _ranking_inputs(predictions, targets, batch):
    if predictions.ndim == 2 and predictions.size(-1) == 1:
        predictions = predictions.squeeze(-1)
    if predictions.ndim != 1 or targets.ndim != 1:
        raise ValueError("Ranking metrics require 1D predictions and targets")
    if predictions.shape != targets.shape:
        raise ValueError("Ranking metrics require aligned prediction and target shapes")
    if batch is None or getattr(batch, "query_index", None) is None:
        raise ValueError("Ranking metrics require batch.query_index")
    query_index = batch.query_index
    if query_index.ndim != 1 or query_index.size(0) != targets.size(0):
        raise ValueError("Ranking metrics require batch.query_index aligned with predictions")
    return predictions, targets, query_index


def _positive_rank(predictions, targets):
    positive_mask = targets > 0.5
    positive_count = int(positive_mask.sum().item())
    if positive_count != 1:
        raise ValueError("Ranking metrics require exactly one positive target per query")
    positive_score = predictions[positive_mask][0]
    return 1 + int((predictions[~positive_mask] > positive_score).sum().item())


def _filtered_rank_inputs(predictions, targets, batch):
    predictions, targets, query_index = _ranking_inputs(predictions, targets, batch)
    filter_mask = getattr(batch, "filter_mask", None)
    if filter_mask is None:
        raise ValueError("Filtered ranking metrics require batch.filter_mask")
    if filter_mask.ndim != 1 or filter_mask.size(0) != targets.size(0):
        raise ValueError("Filtered ranking metrics require batch.filter_mask aligned with predictions")
    return predictions, targets, query_index, filter_mask


def _ordered_unique_query_ids(query_index):
    return tuple(dict.fromkeys(int(query_id) for query_id in query_index.detach().cpu().tolist()))


class MRR(Metric):
    name = "mrr"

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reciprocal_rank = 0.0
        self.total_queries = 0

    def update(self, predictions, targets, *, batch=None):
        predictions, targets, query_index = _ranking_inputs(predictions, targets, batch)
        for query_id in _ordered_unique_query_ids(query_index):
            mask = query_index == query_id
            rank = _positive_rank(predictions[mask], targets[mask])
            self.total_reciprocal_rank += 1.0 / rank
            self.total_queries += 1

    def compute(self):
        if self.total_queries == 0:
            raise ValueError("MRR requires at least one query before compute()")
        return self.total_reciprocal_rank / self.total_queries


class HitsAtK(Metric):
    def __init__(self, k):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.name = f"hits@{self.k}"
        self.reset()

    def reset(self):
        self.total_hits = 0
        self.total_queries = 0

    def update(self, predictions, targets, *, batch=None):
        predictions, targets, query_index = _ranking_inputs(predictions, targets, batch)
        for query_id in _ordered_unique_query_ids(query_index):
            mask = query_index == query_id
            rank = _positive_rank(predictions[mask], targets[mask])
            self.total_hits += int(rank <= self.k)
            self.total_queries += 1

    def compute(self):
        if self.total_queries == 0:
            raise ValueError(f"{self.name} requires at least one query before compute()")
        return self.total_hits / self.total_queries


class FilteredMRR(Metric):
    name = "filtered_mrr"

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_reciprocal_rank = 0.0
        self.total_queries = 0

    def update(self, predictions, targets, *, batch=None):
        predictions, targets, query_index, filter_mask = _filtered_rank_inputs(predictions, targets, batch)
        for query_id in _ordered_unique_query_ids(query_index):
            mask = query_index == query_id
            active_mask = mask & ~filter_mask
            rank = _positive_rank(predictions[active_mask], targets[active_mask])
            self.total_reciprocal_rank += 1.0 / rank
            self.total_queries += 1

    def compute(self):
        if self.total_queries == 0:
            raise ValueError("FilteredMRR requires at least one query before compute()")
        return self.total_reciprocal_rank / self.total_queries


class FilteredHitsAtK(Metric):
    def __init__(self, k):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.k = int(k)
        self.name = f"filtered_hits@{self.k}"
        self.reset()

    def reset(self):
        self.total_hits = 0
        self.total_queries = 0

    def update(self, predictions, targets, *, batch=None):
        predictions, targets, query_index, filter_mask = _filtered_rank_inputs(predictions, targets, batch)
        for query_id in _ordered_unique_query_ids(query_index):
            mask = query_index == query_id
            active_mask = mask & ~filter_mask
            rank = _positive_rank(predictions[active_mask], targets[active_mask])
            self.total_hits += int(rank <= self.k)
            self.total_queries += 1

    def compute(self):
        if self.total_queries == 0:
            raise ValueError(f"{self.name} requires at least one query before compute()")
        return self.total_hits / self.total_queries


__all__ = ["FilteredHitsAtK", "FilteredMRR", "HitsAtK", "MRR"]
