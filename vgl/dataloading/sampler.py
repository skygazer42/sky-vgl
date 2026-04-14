from dataclasses import replace

import torch

from vgl._neighbor_sampling import (
    _exclude_members,
    _merge_sorted_unique_tensors,
    _relation_neighbor_candidates,
    _sample_fanout,
    _sample_homo_next_frontier,
    _sample_homo_node_ids,
    _sorted_unique_tensor,
)
from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.materialize import materialize_context
from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.records import LinkPredictionRecord
from vgl.dataloading.records import SampleRecord
from vgl.dataloading.records import TemporalEventRecord
from vgl.dataloading.requests import LinkSeedRequest, NodeSeedRequest, TemporalSeedRequest
from vgl.graph.graph import Graph
from vgl.ops.subgraph import (
    _expand_interval_values,
    _lookup_positions,
    _membership_mask,
    _positions_for_endpoint_values,
    _relabel_bipartite_edge_index,
    _slice_edge_store,
    node_subgraph,
)


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _resolve_link_edge_type(record):
    edge_type = getattr(record, "edge_type", None)
    if edge_type is None:
        edge_type = record.metadata.get("edge_type")
    if edge_type is not None:
        return tuple(edge_type)
    graph = record.graph
    if len(graph.edges) == 1:
        return next(iter(graph.edges))
    try:
        return graph._default_edge_type()
    except AttributeError as exc:
        raise ValueError("Link prediction on heterogeneous graphs requires edge_type") from exc


def _link_endpoint_types(record):
    edge_type = _resolve_link_edge_type(record)
    return edge_type, edge_type[0], edge_type[2]


def _resolve_link_reverse_edge_type(record):
    reverse_edge_type = getattr(record, "reverse_edge_type", None)
    if reverse_edge_type is None:
        reverse_edge_type = record.metadata.get("reverse_edge_type")
    if reverse_edge_type is None:
        return None
    return tuple(reverse_edge_type)


def _single_link_edge_type(records, *, context: str) -> tuple[str, str, str] | None:
    edge_types = tuple(dict.fromkeys(_resolve_link_edge_type(record) for record in records))
    if len(edge_types) == 1:
        return edge_types[0]
    return None


def _single_inbound_node_block_edge_type(graph: Graph, *, node_type: str, context: str) -> tuple[str, str, str] | None:
    if set(graph.nodes) == {"node"} and len(graph.edges) == 1:
        return graph._default_edge_type()

    inbound_edge_types = tuple(edge_type for edge_type in graph.edges if edge_type[2] == node_type)
    if len(inbound_edge_types) == 1:
        return inbound_edge_types[0]
    return None


def _history_edge_positions(history_edge_index: torch.Tensor, src_node_ids: torch.Tensor, dst_node_ids: torch.Tensor) -> torch.Tensor:
    history_edge_index = torch.as_tensor(history_edge_index, dtype=torch.long)
    if history_edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=history_edge_index.device)

    class _HistoryStore:
        def __init__(self, edge_index):
            self.edge_index = edge_index
            self.query_cache = {}

    store = _HistoryStore(history_edge_index)
    src_node_ids = torch.as_tensor(src_node_ids, dtype=torch.long, device=history_edge_index.device).view(-1)
    dst_node_ids = torch.as_tensor(dst_node_ids, dtype=torch.long, device=history_edge_index.device).view(-1)
    candidate_positions = _positions_for_endpoint_values(store, src_node_ids, endpoint=0)
    if candidate_positions.numel() == 0:
        return candidate_positions
    return candidate_positions[
        _membership_mask(history_edge_index[1, candidate_positions], dst_node_ids)
    ]


def _history_neighbor_candidates(edge_store, history_edge_ids: torch.Tensor, frontier, *, endpoint: int) -> torch.Tensor:
    history_edge_ids = torch.as_tensor(history_edge_ids, dtype=torch.long, device=edge_store.edge_index.device).view(-1)
    frontier = torch.as_tensor(frontier, dtype=torch.long, device=edge_store.edge_index.device).view(-1)
    if history_edge_ids.numel() == 0 or frontier.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_store.edge_index.device)

    candidate_edge_ids = _positions_for_endpoint_values(edge_store, frontier, endpoint=endpoint)
    if candidate_edge_ids.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_store.edge_index.device)
    kept = _sorted_membership_mask(candidate_edge_ids, history_edge_ids)
    if not bool(kept.any()):
        return torch.empty(0, dtype=torch.long, device=edge_store.edge_index.device)
    return edge_store.edge_index[1 - endpoint, candidate_edge_ids[kept]]


def _sample_homo_history_node_ids(
    edge_store,
    history_edge_ids: torch.Tensor,
    seed_nodes,
    num_neighbors,
    *,
    generator,
) -> torch.Tensor:
    edge_index = edge_store.edge_index
    visited = _sorted_unique_tensor(
        torch.as_tensor(seed_nodes, dtype=torch.long, device=edge_index.device).view(-1)
    )
    frontier = visited
    for fanout in num_neighbors:
        neighbor_chunks = []
        src_neighbors = _history_neighbor_candidates(edge_store, history_edge_ids, frontier, endpoint=0)
        if src_neighbors.numel() > 0:
            neighbor_chunks.append(src_neighbors)
        dst_neighbors = _history_neighbor_candidates(edge_store, history_edge_ids, frontier, endpoint=1)
        if dst_neighbors.numel() > 0:
            neighbor_chunks.append(dst_neighbors)
        if not neighbor_chunks:
            break

        frontier = _sample_fanout(
            _exclude_members(torch.cat(neighbor_chunks), visited),
            fanout,
            generator=generator,
        )
        if frontier.numel() == 0:
            break
        visited = _merge_sorted_unique_tensors(visited, frontier)
    return visited


def _history_relation_neighbor_candidates(
    edge_store,
    history_edge_ids: torch.Tensor,
    frontier,
    visited,
    *,
    src_type,
    dst_type,
):
    edge_index = edge_store.edge_index
    unique_types = tuple(dict.fromkeys((src_type, dst_type)))
    empty = {
        node_type: torch.empty(0, dtype=torch.long, device=edge_index.device)
        for node_type in unique_types
    }
    history_edge_ids = torch.as_tensor(history_edge_ids, dtype=torch.long, device=edge_index.device).view(-1)
    if history_edge_ids.numel() == 0:
        return empty

    src_frontier = frontier.get(src_type)
    dst_frontier = frontier.get(dst_type)
    if src_type == dst_type:
        candidate_chunks = []
        if src_frontier is not None and src_frontier.numel() > 0:
            src_frontier = src_frontier.to(device=edge_index.device)
            src_candidates = _history_neighbor_candidates(edge_store, history_edge_ids, src_frontier, endpoint=0)
            if src_candidates.numel() > 0:
                candidate_chunks.append(src_candidates)
            dst_candidates = _history_neighbor_candidates(edge_store, history_edge_ids, src_frontier, endpoint=1)
            if dst_candidates.numel() > 0:
                candidate_chunks.append(dst_candidates)
        if not candidate_chunks:
            return empty
        candidates = torch.cat(candidate_chunks)
        visited_tensor = visited.get(src_type)
        if visited_tensor is None:
            visited_tensor = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return {src_type: _exclude_members(candidates, visited_tensor)}

    if dst_frontier is not None and dst_frontier.numel() > 0:
        src_candidates = _history_neighbor_candidates(
            edge_store,
            history_edge_ids,
            dst_frontier.to(device=edge_index.device),
            endpoint=1,
        )
    else:
        src_candidates = torch.empty(0, dtype=torch.long, device=edge_index.device)
    if src_frontier is not None and src_frontier.numel() > 0:
        dst_candidates = _history_neighbor_candidates(
            edge_store,
            history_edge_ids,
            src_frontier.to(device=edge_index.device),
            endpoint=0,
        )
    else:
        dst_candidates = torch.empty(0, dtype=torch.long, device=edge_index.device)

    visited_src = visited.get(src_type)
    if visited_src is None:
        visited_src = torch.empty(0, dtype=torch.long, device=edge_index.device)
    visited_dst = visited.get(dst_type)
    if visited_dst is None:
        visited_dst = torch.empty(0, dtype=torch.long, device=edge_index.device)
    return {
        src_type: _exclude_members(src_candidates, visited_src),
        dst_type: _exclude_members(dst_candidates, visited_dst),
    }


def _history_edge_ids_for_nodes(edge_store, history_edge_ids: torch.Tensor, src_node_ids, dst_node_ids) -> torch.Tensor:
    history_edge_ids = torch.as_tensor(history_edge_ids, dtype=torch.long, device=edge_store.edge_index.device).view(-1)
    if history_edge_ids.numel() == 0:
        return history_edge_ids
    candidate_edge_ids = _positions_for_endpoint_values(edge_store, src_node_ids, endpoint=0)
    if candidate_edge_ids.numel() == 0:
        return candidate_edge_ids
    candidate_edge_ids = candidate_edge_ids[_sorted_membership_mask(candidate_edge_ids, history_edge_ids)]
    if candidate_edge_ids.numel() == 0:
        return candidate_edge_ids
    return candidate_edge_ids[
        _sorted_membership_mask(edge_store.edge_index[1, candidate_edge_ids], dst_node_ids)
    ]


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        int(tensor.data_ptr()) if tensor.numel() > 0 else 0,
        tuple(int(dim) for dim in tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        int(getattr(tensor, "_version", 0)),
    )


def _sorted_membership_mask(values: torch.Tensor, sorted_allowed_ids: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    sorted_allowed_ids = torch.as_tensor(sorted_allowed_ids, dtype=torch.long, device=values.device).view(-1)
    if values.numel() == 0 or sorted_allowed_ids.numel() == 0:
        return torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    lower = sorted_allowed_ids[0]
    upper = sorted_allowed_ids[-1]
    if (upper - lower + 1) == sorted_allowed_ids.numel():
        return (values >= lower) & (values <= upper)
    positions = torch.searchsorted(sorted_allowed_ids, values, right=False)
    capped_positions = positions.clamp_max(sorted_allowed_ids.numel() - 1)
    return (positions < sorted_allowed_ids.numel()) & (sorted_allowed_ids[capped_positions] == values)


def _temporal_history_lookup(edge_store, *, time_attr: str) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if hasattr(edge_store, "query_cache"):
        cache = edge_store.query_cache
    else:
        cache = {}

    cache_key = ("temporal_history_lookup", str(time_attr))
    timestamps = torch.as_tensor(
        edge_store.data[time_attr],
        device=edge_store.edge_index.device,
    ).view(-1)
    signature = _tensor_signature(timestamps)
    entry = cache.get(cache_key)
    if entry is not None and entry["signature"] == signature:
        return entry["sorted_timestamps"], entry["sorted_edge_ids"], entry["edge_ids_are_sorted"]

    edge_ids = torch.arange(timestamps.numel(), dtype=torch.long, device=timestamps.device)
    order = torch.argsort(timestamps, stable=True)
    entry = {
        "signature": signature,
        "sorted_timestamps": timestamps[order],
        "sorted_edge_ids": edge_ids[order],
        "edge_ids_are_sorted": True if order.numel() <= 1 else bool((order[1:] >= order[:-1]).all()),
    }
    cache[cache_key] = entry
    if hasattr(edge_store, "query_cache"):
        edge_store.query_cache = cache
    return entry["sorted_timestamps"], entry["sorted_edge_ids"], entry["edge_ids_are_sorted"]


def _resolve_temporal_edge_type(record):
    edge_type = getattr(record, "edge_type", None)
    if edge_type is None:
        edge_type = record.metadata.get("edge_type")
    if edge_type is not None:
        return tuple(edge_type)
    graph = record.graph
    if len(graph.edges) == 1:
        return next(iter(graph.edges))
    try:
        return graph._default_edge_type()
    except AttributeError as exc:
        raise ValueError("Temporal event prediction on heterogeneous graphs requires edge_type") from exc


def _temporal_endpoint_types(record):
    edge_type = _resolve_temporal_edge_type(record)
    return edge_type, edge_type[0], edge_type[2]


def _ordered_unique_tensor(values) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    if values.numel() == 0:
        return values
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    keep = torch.ones(sorted_values.numel(), dtype=torch.bool, device=values.device)
    if sorted_values.numel() > 1:
        keep[1:] = sorted_values[1:] != sorted_values[:-1]
    first_occurrences = torch.sort(order[keep], stable=True).values
    return values[first_occurrences]


def _available_destinations_from_excluded(num_nodes: int, excluded: torch.Tensor) -> torch.Tensor:
    if excluded.numel() == 0:
        return torch.arange(int(num_nodes), dtype=torch.long, device=excluded.device)

    starts = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=excluded.device),
            excluded + 1,
        )
    )
    ends = torch.cat(
        (
            excluded - 1,
            torch.tensor([int(num_nodes) - 1], dtype=torch.long, device=excluded.device),
        )
    )
    lengths = ends - starts + 1
    keep = lengths > 0
    starts = starts[keep]
    lengths = lengths[keep]
    return _expand_interval_values(starts, lengths, step=1)


def _sample_non_excluded_destinations(
    num_nodes: int,
    excluded_destinations,
    count: int,
    *,
    device: torch.device | None = None,
) -> torch.Tensor:
    device = torch.device(device) if device is not None else None
    excluded = torch.as_tensor(excluded_destinations, dtype=torch.long, device=device).view(-1)
    if excluded.numel() > 0:
        excluded = excluded[(excluded >= 0) & (excluded < int(num_nodes))]
        if excluded.numel() > 0:
            excluded = _sorted_unique_tensor(excluded)

    available_count = int(num_nodes) - int(excluded.numel())
    if available_count <= 0:
        raise ValueError("UniformNegativeLinkSampler could not find a valid negative destination")
    if count <= 0:
        return torch.empty((0,), dtype=torch.long, device=excluded.device)
    if excluded.numel() == 0:
        return torch.randint(int(num_nodes), (int(count),), device=excluded.device)

    if available_count <= max(64, int(count) * 4):
        available = _available_destinations_from_excluded(int(num_nodes), excluded)
        indices = torch.randint(available.numel(), (int(count),), device=excluded.device)
        return available[indices]

    samples = torch.empty((int(count),), dtype=torch.long, device=excluded.device)
    filled = 0
    proposal_size = max(int(count), 32)
    max_attempts = 8
    attempts = 0
    while filled < int(count):
        attempts += 1
        proposed = torch.randint(int(num_nodes), (proposal_size,), device=excluded.device)
        accepted = proposed[~_membership_mask(proposed, excluded)]
        if accepted.numel() == 0:
            if attempts >= max_attempts:
                available = _available_destinations_from_excluded(int(num_nodes), excluded)
                indices = torch.randint(available.numel(), (int(count),), device=excluded.device)
                return available[indices]
            proposal_size = max(proposal_size * 2, int(count) - filled)
            continue
        take = min(int(count) - filled, int(accepted.numel()))
        samples[filled : filled + take] = accepted[:take]
        filled += take

    return samples


class Sampler:
    def sample(self, item):
        raise NotImplementedError


class FullGraphSampler(Sampler):
    def sample(self, graph):
        return graph


class UniformNegativeLinkSampler(Sampler):
    def __init__(
        self,
        num_negatives=1,
        *,
        exclude_positive_edges=True,
        exclude_seed_edges=False,
        skip_negative_seed_records: bool = False,
    ):
        resolved_num_negatives = _as_python_int(num_negatives)
        if resolved_num_negatives < 1:
            raise ValueError("num_negatives must be >= 1")
        self.num_negatives = resolved_num_negatives
        self.exclude_positive_edges = bool(exclude_positive_edges)
        self.exclude_seed_edges = bool(exclude_seed_edges)
        self.skip_negative_seed_records = bool(skip_negative_seed_records)

    def _candidate_destinations(self, item):
        edge_type, _, dst_type = _link_endpoint_types(item)
        graph = item.graph
        num_nodes = int(graph.nodes[dst_type].x.size(0))
        mask = torch.ones(num_nodes, dtype=torch.bool)
        mask[_as_python_int(item.dst_index)] = False
        if self.exclude_positive_edges:
            edge_index = graph.edges[edge_type].edge_index
            positive_mask = edge_index[0] == _as_python_int(item.src_index)
            if positive_mask.any():
                mask[edge_index[1, positive_mask].to(dtype=torch.long)] = False
        return torch.arange(num_nodes, dtype=torch.long)[mask]

    def _excluded_negative_destinations(self, item):
        edge_type, _, dst_type = _link_endpoint_types(item)
        graph = item.graph
        edge_index = graph.edges[edge_type].edge_index
        device = edge_index.device
        excluded = torch.tensor([_as_python_int(item.dst_index)], dtype=torch.long, device=device)
        if not self.exclude_positive_edges:
            return excluded
        positive_mask = edge_index[0] == _as_python_int(item.src_index)
        if not bool(positive_mask.any()):
            return excluded
        positive_destinations = edge_index[1, positive_mask].to(dtype=torch.long)
        return _merge_sorted_unique_tensors(excluded, positive_destinations)

    def _resolved_query_id(self, item):
        query_id = getattr(item, "resolved_query_id", None)
        if query_id is not None:
            return query_id
        query_id = item.query_id
        if query_id is None:
            query_id = item.metadata.get("query_id")
        if query_id is None:
            query_id = self._resolved_sample_id(item)
        if query_id is None:
            query_id = id(item)
        return query_id

    def _resolved_sample_id(self, item):
        sample_id = getattr(item, "resolved_sample_id", None)
        if sample_id is not None:
            return sample_id
        sample_id = item.sample_id
        if sample_id is None:
            sample_id = item.metadata.get("sample_id")
        return sample_id

    def _resolved_hard_negative_dst(self, item):
        raw_candidates = item.hard_negative_dst
        if raw_candidates is None:
            raw_candidates = item.metadata.get("hard_negative_dst")
        return raw_candidates

    def _resolved_candidate_dst(self, item):
        raw_candidates = item.candidate_dst
        if raw_candidates is None:
            raw_candidates = item.metadata.get("candidate_dst")
        return raw_candidates

    def _resolved_exclude_seed_edge(self, item):
        return bool(item.exclude_seed_edge) or bool(item.metadata.get("exclude_seed_edges", False)) or self.exclude_seed_edges

    def _positive_seed_record(self, item):
        sample_id = self._resolved_sample_id(item)
        query_id = self._resolved_query_id(item)
        hard_negative_dst = self._resolved_hard_negative_dst(item)
        candidate_dst = self._resolved_candidate_dst(item)
        edge_type = _resolve_link_edge_type(item)
        reverse_edge_type = _resolve_link_reverse_edge_type(item)
        exclude_seed_edge = self._resolved_exclude_seed_edge(item)
        metadata = dict(item.metadata)
        if sample_id is not None:
            metadata["sample_id"] = sample_id
        if query_id is not None:
            metadata["query_id"] = query_id
        metadata["edge_type"] = edge_type
        if reverse_edge_type is not None:
            metadata["reverse_edge_type"] = reverse_edge_type
        if hard_negative_dst is not None:
            metadata["hard_negative_dst"] = hard_negative_dst
        if candidate_dst is not None:
            metadata["candidate_dst"] = candidate_dst
        if exclude_seed_edge:
            metadata["exclude_seed_edges"] = True
        return LinkPredictionRecord(
            graph=item.graph,
            src_index=_as_python_int(item.src_index),
            dst_index=_as_python_int(item.dst_index),
            label=1,
            metadata=metadata,
            sample_id=sample_id,
            exclude_seed_edge=exclude_seed_edge,
            hard_negative_dst=hard_negative_dst,
            candidate_dst=candidate_dst,
            edge_type=edge_type,
            reverse_edge_type=reverse_edge_type,
            query_id=query_id,
        )

    def _negative_record(self, item, dst_index, offset, *, hard_negative=False, query_id=None):
        resolved_sample_id = self._resolved_sample_id(item)
        hard_negative_dst = self._resolved_hard_negative_dst(item)
        candidate_dst = self._resolved_candidate_dst(item)
        edge_type = _resolve_link_edge_type(item)
        reverse_edge_type = _resolve_link_reverse_edge_type(item)
        negative_metadata = dict(item.metadata)
        negative_metadata.pop("exclude_seed_edges", None)
        negative_metadata["negative_sampled"] = True
        if hard_negative:
            negative_metadata["hard_negative_sampled"] = True
        suffix = "hard-neg" if hard_negative else "neg"
        sample_id = None if resolved_sample_id is None else f"{resolved_sample_id}:{suffix}:{offset}"
        if sample_id is not None:
            negative_metadata["sample_id"] = sample_id
        if query_id is not None:
            negative_metadata["query_id"] = query_id
        negative_metadata["edge_type"] = edge_type
        if reverse_edge_type is not None:
            negative_metadata["reverse_edge_type"] = reverse_edge_type
        if hard_negative_dst is not None:
            negative_metadata["hard_negative_dst"] = hard_negative_dst
        if candidate_dst is not None:
            negative_metadata["candidate_dst"] = candidate_dst
        return LinkPredictionRecord(
            graph=item.graph,
            src_index=_as_python_int(item.src_index),
            dst_index=_as_python_int(dst_index),
            label=0,
            metadata=negative_metadata,
            sample_id=sample_id,
            exclude_seed_edge=False,
            hard_negative_dst=hard_negative_dst,
            candidate_dst=candidate_dst,
            edge_type=edge_type,
            reverse_edge_type=reverse_edge_type,
            query_id=query_id,
        )

    def _uniform_destinations(self, item, count, *, excluded_destinations=()):
        edge_type, _, dst_type = _link_endpoint_types(item)
        num_nodes = int(item.graph.nodes[dst_type].x.size(0))
        device = item.graph.edges[edge_type].edge_index.device
        excluded = self._excluded_negative_destinations(item)
        additional_excluded = torch.as_tensor(excluded_destinations, dtype=torch.long, device=device).view(-1)
        if additional_excluded.numel() > 0:
            excluded = torch.cat((excluded, additional_excluded))
        return _sample_non_excluded_destinations(num_nodes, excluded, int(count), device=device)

    def sample(self, item):
        if not isinstance(item, LinkPredictionRecord):
            raise TypeError("UniformNegativeLinkSampler requires LinkPredictionRecord items")
        label = _as_python_int(item.label)
        if label != 1:
            if self.skip_negative_seed_records and label == 0:
                return ()
            raise ValueError("UniformNegativeLinkSampler requires positive seed records")
        positive = self._positive_seed_record(item)
        sampled = [positive]
        destinations = self._uniform_destinations(item, self.num_negatives)
        for offset, dst_index in enumerate(destinations.view(-1)):
            sampled.append(self._negative_record(item, _as_python_int(dst_index), offset, query_id=positive.query_id))
        return sampled


class HardNegativeLinkSampler(UniformNegativeLinkSampler):
    def __init__(
        self,
        num_negatives=1,
        *,
        num_hard_negatives=1,
        exclude_positive_edges=True,
        exclude_seed_edges=False,
        skip_negative_seed_records: bool = False,
        hard_negative_dst_metadata_key: str | None = None,
    ):
        super().__init__(
            num_negatives=num_negatives,
            exclude_positive_edges=exclude_positive_edges,
            exclude_seed_edges=exclude_seed_edges,
            skip_negative_seed_records=skip_negative_seed_records,
        )
        resolved_num_hard_negatives = _as_python_int(num_hard_negatives)
        if resolved_num_hard_negatives < 0:
            raise ValueError("num_hard_negatives must be >= 0")
        self.num_hard_negatives = resolved_num_hard_negatives
        self.hard_negative_dst_metadata_key = hard_negative_dst_metadata_key

    def _resolved_hard_negative_dst(self, item):
        raw_candidates = item.hard_negative_dst
        if raw_candidates is None and self.hard_negative_dst_metadata_key is not None:
            raw_candidates = item.metadata.get(self.hard_negative_dst_metadata_key)
        if raw_candidates is None:
            raw_candidates = item.metadata.get("hard_negative_dst")
        return raw_candidates

    def _hard_negative_candidates(self, item):
        raw_candidates = self._resolved_hard_negative_dst(item)
        if raw_candidates is None:
            return torch.empty((0,), dtype=torch.long)
        candidates = torch.as_tensor(raw_candidates, dtype=torch.long).view(-1)
        _, _, dst_type = _link_endpoint_types(item)
        num_nodes = int(item.graph.nodes[dst_type].x.size(0))
        if ((candidates < 0) | (candidates >= num_nodes)).any():
            raise ValueError("hard_negative_dst entries must fall within the source graph node range")
        candidates = _ordered_unique_tensor(candidates)
        excluded = self._excluded_negative_destinations(item).to(device=candidates.device)
        if excluded.numel() == 0:
            return candidates
        return candidates[~_membership_mask(candidates, excluded)]

    def sample(self, item):
        if not isinstance(item, LinkPredictionRecord):
            raise TypeError("HardNegativeLinkSampler requires LinkPredictionRecord items")
        label = _as_python_int(item.label)
        if label != 1:
            if self.skip_negative_seed_records and label == 0:
                return ()
            raise ValueError("HardNegativeLinkSampler requires positive seed records")

        positive = self._positive_seed_record(item)
        sampled = [positive]
        hard_candidates = self._hard_negative_candidates(item)
        hard_count = min(self.num_negatives, self.num_hard_negatives, int(hard_candidates.numel()))
        selected_hard = torch.empty((0,), dtype=torch.long, device=hard_candidates.device)
        if hard_count > 0:
            permutation = torch.randperm(hard_candidates.numel(), device=hard_candidates.device)[:hard_count]
            selected_hard = hard_candidates[permutation]
            for offset, dst_index in enumerate(selected_hard.view(-1)):
                sampled.append(
                    self._negative_record(
                        item,
                        dst_index,
                        offset,
                        hard_negative=True,
                        query_id=positive.query_id,
                    )
                )

        remaining = self.num_negatives - int(selected_hard.numel())
        if remaining > 0:
            uniform_destinations = self._uniform_destinations(
                item,
                remaining,
                excluded_destinations=selected_hard,
            )
            start_offset = int(selected_hard.numel())
            for offset, dst_index in enumerate(uniform_destinations.view(-1), start=start_offset):
                sampled.append(self._negative_record(item, dst_index, offset, query_id=positive.query_id))
        return sampled


class CandidateLinkSampler(UniformNegativeLinkSampler):
    def __init__(
        self,
        *,
        filter_known_positive_edges=True,
        exclude_seed_edges=False,
        skip_negative_seed_records: bool = True,
        candidate_dst_metadata_key: str | None = None,
    ):
        super().__init__(
            num_negatives=1,
            exclude_positive_edges=False,
            exclude_seed_edges=exclude_seed_edges,
            skip_negative_seed_records=skip_negative_seed_records,
        )
        self.filter_known_positive_edges = bool(filter_known_positive_edges)
        self.candidate_dst_metadata_key = candidate_dst_metadata_key

    def _resolved_candidate_dst(self, item):
        raw_candidates = item.candidate_dst
        if raw_candidates is None and self.candidate_dst_metadata_key is not None:
            raw_candidates = item.metadata.get(self.candidate_dst_metadata_key)
        if raw_candidates is None:
            raw_candidates = item.metadata.get("candidate_dst")
        return raw_candidates

    def _candidate_destinations(self, item):
        raw_candidates = self._resolved_candidate_dst(item)
        _, _, dst_type = _link_endpoint_types(item)
        num_nodes = int(item.graph.nodes[dst_type].x.size(0))
        if raw_candidates is None:
            return torch.arange(num_nodes, dtype=torch.long)
        candidates = torch.as_tensor(raw_candidates, dtype=torch.long).view(-1)
        if ((candidates < 0) | (candidates >= num_nodes)).any():
            raise ValueError("candidate_dst entries must fall within the source graph node range")
        return _ordered_unique_tensor(candidates)

    def _known_positive_destinations(self, item):
        edge_type = _resolve_link_edge_type(item)
        edge_index = item.graph.edges[edge_type].edge_index
        positive_mask = edge_index[0] == _as_python_int(item.src_index)
        if not positive_mask.any():
            return torch.empty((0,), dtype=torch.long, device=edge_index.device)
        return _ordered_unique_tensor(edge_index[1, positive_mask].to(dtype=torch.long))

    def sample(self, item):
        if not isinstance(item, LinkPredictionRecord):
            raise TypeError("CandidateLinkSampler requires LinkPredictionRecord items")
        label = _as_python_int(item.label)
        if label != 1:
            if self.skip_negative_seed_records and label == 0:
                return ()
            raise ValueError("CandidateLinkSampler requires positive seed records")

        positive = self._positive_seed_record(item)
        candidates = self._candidate_destinations(item)
        positive_dst_value = _as_python_int(positive.dst_index)
        positive_dst = torch.tensor([positive_dst_value], dtype=torch.long, device=candidates.device)
        ordered_destinations = torch.cat((positive_dst, candidates[candidates != positive_dst_value]))
        known_positive_destinations = torch.empty((0,), dtype=torch.long, device=ordered_destinations.device)
        if self.filter_known_positive_edges:
            known_positive_destinations = self._known_positive_destinations(item)
        filter_mask = (
            _membership_mask(ordered_destinations[1:], known_positive_destinations)
            if ordered_destinations.numel() > 1 and known_positive_destinations.numel() > 0
            else torch.zeros(max(int(ordered_destinations.numel()) - 1, 0), dtype=torch.bool, device=ordered_destinations.device)
        )

        sampled = [positive]
        for offset, dst_index in enumerate(ordered_destinations[1:].view(-1)):
            negative = self._negative_record(item, dst_index, offset, query_id=positive.query_id)
            if bool(filter_mask[offset]):
                negative.filter_ranking = True
                negative.metadata["filter_ranking"] = True
            sampled.append(negative)
        return sampled


class LinkNeighborSampler(Sampler):
    def __init__(
        self,
        num_neighbors,
        *,
        base_sampler=None,
        seed=None,
        node_feature_names=None,
        edge_feature_names=None,
        output_blocks: bool = False,
        replace: bool = False,
        directed: bool = False,
        candidate_dst_metadata_key: str | None = None,
    ):
        if isinstance(num_neighbors, int) or (
            isinstance(num_neighbors, torch.Tensor) and num_neighbors.ndim == 0
        ):
            num_neighbors = [num_neighbors]
        self.num_neighbors = [_as_python_int(value) for value in num_neighbors]
        if not self.num_neighbors:
            raise ValueError("num_neighbors must contain at least one hop")
        if any(value < -1 or value == 0 for value in self.num_neighbors):
            raise ValueError("num_neighbors entries must be positive integers or -1")
        self.base_sampler = base_sampler
        self.node_feature_names = node_feature_names
        self.edge_feature_names = edge_feature_names
        self.output_blocks = bool(output_blocks)
        self.replace = bool(replace)
        self.directed = bool(directed)
        self.candidate_dst_metadata_key = candidate_dst_metadata_key
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(_as_python_int(seed))

    @staticmethod
    def _normalize_feature_names(feature_names):
        return tuple(str(name) for name in feature_names)

    def _resolved_node_feature_names(self, graph):
        if self.node_feature_names is None:
            return ()
        if isinstance(self.node_feature_names, dict):
            resolved = []
            for node_type, feature_names in self.node_feature_names.items():
                node_type = str(node_type)
                if node_type not in graph.nodes:
                    raise ValueError(f"unknown node_type for feature prefetch: {node_type!r}")
                names = self._normalize_feature_names(feature_names)
                if names:
                    resolved.append((node_type, names))
            return tuple(resolved)
        if set(graph.nodes) != {"node"} or len(graph.edges) != 1:
            raise ValueError("heterogeneous LinkNeighborSampler requires node_feature_names keyed by node type")
        names = self._normalize_feature_names(self.node_feature_names)
        if not names:
            return ()
        return (("node", names),)

    def _resolved_edge_feature_names(self, graph):
        if self.edge_feature_names is None:
            return ()
        if isinstance(self.edge_feature_names, dict):
            resolved = []
            for edge_type, feature_names in self.edge_feature_names.items():
                edge_type = tuple(edge_type)
                if edge_type not in graph.edges:
                    raise ValueError(f"unknown edge_type for feature prefetch: {edge_type!r}")
                names = self._normalize_feature_names(feature_names)
                if names:
                    resolved.append((edge_type, names))
            return tuple(resolved)
        if set(graph.nodes) != {"node"} or len(graph.edges) != 1:
            raise ValueError("heterogeneous LinkNeighborSampler requires edge_feature_names keyed by edge type")
        names = self._normalize_feature_names(self.edge_feature_names)
        if not names:
            return ()
        return ((graph._default_edge_type(), names),)

    def _seed_records(self, item):
        if isinstance(item, LinkPredictionRecord):
            item = self._record_with_candidate_pool(item)
        sampled = self.base_sampler.sample(item) if self.base_sampler is not None else item
        is_sequence = isinstance(sampled, (list, tuple))
        records = list(sampled) if is_sequence else [sampled]
        if not records:
            return records, is_sequence
        if any(not isinstance(record, LinkPredictionRecord) for record in records):
            raise TypeError("LinkNeighborSampler requires LinkPredictionRecord items")
        graph = records[0].graph
        if any(record.graph is not graph for record in records):
            raise ValueError("LinkNeighborSampler requires records from a single source graph")
        return records, is_sequence

    def _record_with_candidate_pool(self, record: LinkPredictionRecord) -> LinkPredictionRecord:
        if record.candidate_dst is not None or self.candidate_dst_metadata_key is None:
            return record
        if self.candidate_dst_metadata_key not in record.metadata:
            return record
        metadata = dict(record.metadata)
        metadata["candidate_dst"] = metadata[self.candidate_dst_metadata_key]
        return replace(record, candidate_dst=metadata["candidate_dst"], metadata=metadata)

    def _next_frontier(self, graph, frontier, visited, fanout):
        frontier_tensor = _sample_homo_next_frontier(
            graph.edge_index,
            torch.tensor(sorted(frontier), dtype=torch.long, device=graph.edge_index.device),
            torch.tensor(sorted(visited), dtype=torch.long, device=graph.edge_index.device),
            fanout,
            generator=self._generator,
        )
        return {_as_python_int(node) for node in frontier_tensor}

    def _sample_node_ids(self, graph, records, *, return_hops: bool = False):
        seed_nodes = torch.tensor(
            [
                _as_python_int(node)
                for record in records
                for node in (record.src_index, record.dst_index)
            ],
            dtype=torch.long,
            device=graph.edge_index.device,
        )
        return _sample_homo_node_ids(
            graph.edge_index,
            seed_nodes,
            self.num_neighbors,
            generator=self._generator,
            return_hops=return_hops,
        )

    def _hetero_next_frontier(self, graph, frontier, visited, fanout):
        candidates = {node_type: [] for node_type in graph.schema.node_types}
        for edge_type, store in graph.edges.items():
            src_type, _, dst_type = edge_type
            src_frontier = frontier.get(src_type)
            dst_frontier = frontier.get(dst_type)
            if (src_frontier is None or src_frontier.numel() == 0) and (
                dst_frontier is None or dst_frontier.numel() == 0
            ):
                continue
            relation_candidates = _relation_neighbor_candidates(
                store.edge_index,
                frontier,
                visited,
                src_type=src_type,
                dst_type=dst_type,
            )
            for node_type, node_ids in relation_candidates.items():
                if node_ids.numel() > 0:
                    candidates[node_type].append(node_ids)

        next_frontier = {}
        for node_type, values in candidates.items():
            if not values:
                continue
            candidate_tensor = values[0] if len(values) == 1 else _sorted_unique_tensor(torch.cat(values))
            candidate_tensor = _sample_fanout(candidate_tensor, fanout, generator=self._generator)
            if candidate_tensor.numel() > 0:
                next_frontier[node_type] = candidate_tensor
        return next_frontier

    def _hetero_sample_node_ids(self, graph, records, *, return_hops: bool = False):
        def _snapshot(visited_by_type):
            return {
                node_type: node_ids.clone()
                for node_type, node_ids in visited_by_type.items()
            }

        visited = {
            node_type: torch.empty(
                0,
                dtype=torch.long,
                device=next(iter(graph.nodes[node_type].data.values())).device,
            )
            for node_type in graph.schema.node_types
        }
        seed_values_by_type: dict[str, list[int]] = {node_type: [] for node_type in graph.schema.node_types}
        for record in records:
            edge_type, src_type, dst_type = _link_endpoint_types(record)
            if edge_type not in graph.edges:
                raise ValueError("LinkNeighborSampler record edge_type must exist in the source graph")
            seed_values_by_type[src_type].append(_as_python_int(record.src_index))
            seed_values_by_type[dst_type].append(_as_python_int(record.dst_index))
        for node_type, values in seed_values_by_type.items():
            if values:
                visited[node_type] = _sorted_unique_tensor(
                    torch.tensor(values, dtype=torch.long, device=visited[node_type].device)
                )
        frontier = {
            node_type: node_ids.clone()
            for node_type, node_ids in visited.items()
            if node_ids.numel() > 0
        }
        hop_nodes = [_snapshot(visited)] if return_hops else None
        for fanout in self.num_neighbors:
            frontier = self._hetero_next_frontier(graph, frontier, visited, fanout)
            for node_type, node_ids in frontier.items():
                visited[node_type] = _merge_sorted_unique_tensors(visited[node_type], node_ids)
            if hop_nodes is not None:
                hop_nodes.append(_snapshot(visited))
            elif not frontier:
                break
        sampled = {node_type: node_ids.clone() for node_type, node_ids in visited.items()}
        if hop_nodes is not None:
            return sampled, hop_nodes
        return sampled

    def _subgraph(self, graph, node_ids):
        node_ids = node_ids.to(dtype=torch.long)
        subgraph = node_subgraph(graph, node_ids)
        if "n_id" not in subgraph.nodes["node"].data:
            node_data = dict(subgraph.nodes["node"].data)
            node_data["n_id"] = node_ids
            edge_store = subgraph.edges[subgraph._default_edge_type()]
            edge_data = {
                key: value
                for key, value in edge_store.data.items()
                if key != "edge_index"
            }
            subgraph = Graph.homo(edge_index=edge_store.edge_index, edge_data=edge_data, **node_data)
        return subgraph, node_ids

    def _hetero_subgraph(self, graph, node_ids_by_type):
        nodes = {}
        for node_type, store in graph.nodes.items():
            node_ids = node_ids_by_type[node_type].to(dtype=torch.long)
            num_nodes = store.x.size(0)

            node_data = {}
            for key, value in store.data.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == num_nodes:
                    node_data[key] = value[node_ids]
                else:
                    node_data[key] = value
            if "n_id" not in node_data:
                node_data["n_id"] = node_ids
            nodes[node_type] = node_data

        edges = {}
        for edge_type, store in graph.edges.items():
            src_type, _, dst_type = edge_type
            src_node_ids = node_ids_by_type[src_type].to(dtype=torch.long, device=store.edge_index.device)
            dst_node_ids = node_ids_by_type[dst_type].to(dtype=torch.long, device=store.edge_index.device)
            candidate_edge_ids = _positions_for_endpoint_values(store, src_node_ids, endpoint=0)
            edge_ids = candidate_edge_ids[
                _membership_mask(store.edge_index[1, candidate_edge_ids], dst_node_ids)
            ]
            edge_data = _slice_edge_store(store, edge_ids)
            edge_data["edge_index"] = _relabel_bipartite_edge_index(
                edge_data["edge_index"],
                node_ids_by_type[src_type],
                node_ids_by_type[dst_type],
            )
            if "e_id" not in edge_data:
                edge_data["e_id"] = edge_ids
            edges[edge_type] = edge_data
        return Graph.hetero(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr), {
            node_type: node_ids_by_type[node_type].to(dtype=torch.long)
            for node_type in graph.nodes
        }

    def _local_record(self, record, graph, node_mapping):
        metadata = dict(record.metadata)
        edge_type = _resolve_link_edge_type(record)
        reverse_edge_type = _resolve_link_reverse_edge_type(record)
        if record.sample_id is not None:
            metadata["sample_id"] = record.sample_id
        if record.query_id is not None:
            metadata["query_id"] = record.query_id
        metadata["edge_type"] = edge_type
        if reverse_edge_type is not None:
            metadata["reverse_edge_type"] = reverse_edge_type
        if record.hard_negative_dst is not None:
            metadata["hard_negative_dst"] = record.hard_negative_dst
        if record.candidate_dst is not None:
            metadata["candidate_dst"] = record.candidate_dst
        if bool(record.exclude_seed_edge):
            metadata["exclude_seed_edges"] = True
        if bool(record.filter_ranking):
            metadata["filter_ranking"] = True
        local_positions = _lookup_positions(
            node_mapping,
            torch.tensor(
                [_as_python_int(record.src_index), _as_python_int(record.dst_index)],
                dtype=torch.long,
                device=node_mapping.device,
            ),
            entity_name="node",
        )
        return LinkPredictionRecord(
            graph=graph,
            src_index=_as_python_int(local_positions[0]),
            dst_index=_as_python_int(local_positions[1]),
            label=_as_python_int(record.label),
            metadata=metadata,
            sample_id=record.sample_id,
            exclude_seed_edge=bool(record.exclude_seed_edge),
            hard_negative_dst=record.hard_negative_dst,
            candidate_dst=record.candidate_dst,
            edge_type=edge_type,
            reverse_edge_type=reverse_edge_type,
            query_id=record.query_id,
            filter_ranking=bool(record.filter_ranking),
        )

    def _hetero_local_record(self, record, graph, node_mapping):
        edge_type, src_type, dst_type = _link_endpoint_types(record)
        reverse_edge_type = _resolve_link_reverse_edge_type(record)
        metadata = dict(record.metadata)
        if record.sample_id is not None:
            metadata["sample_id"] = record.sample_id
        if record.query_id is not None:
            metadata["query_id"] = record.query_id
        metadata["edge_type"] = edge_type
        if reverse_edge_type is not None:
            metadata["reverse_edge_type"] = reverse_edge_type
        if record.hard_negative_dst is not None:
            metadata["hard_negative_dst"] = record.hard_negative_dst
        if record.candidate_dst is not None:
            metadata["candidate_dst"] = record.candidate_dst
        if bool(record.exclude_seed_edge):
            metadata["exclude_seed_edges"] = True
        if bool(record.filter_ranking):
            metadata["filter_ranking"] = True
        src_position = _lookup_positions(
            node_mapping[src_type],
            torch.tensor(
                [_as_python_int(record.src_index)],
                dtype=torch.long,
                device=node_mapping[src_type].device,
            ),
            entity_name="source node",
        )
        dst_position = _lookup_positions(
            node_mapping[dst_type],
            torch.tensor(
                [_as_python_int(record.dst_index)],
                dtype=torch.long,
                device=node_mapping[dst_type].device,
            ),
            entity_name="destination node",
        )
        return LinkPredictionRecord(
            graph=graph,
            src_index=_as_python_int(src_position[0]),
            dst_index=_as_python_int(dst_position[0]),
            label=_as_python_int(record.label),
            metadata=metadata,
            sample_id=record.sample_id,
            exclude_seed_edge=bool(record.exclude_seed_edge),
            hard_negative_dst=record.hard_negative_dst,
            candidate_dst=record.candidate_dst,
            edge_type=edge_type,
            reverse_edge_type=reverse_edge_type,
            query_id=record.query_id,
            filter_ranking=bool(record.filter_ranking),
        )

    def _sample_from_seed_records(self, records, *, is_sequence, return_hops: bool = False):
        graph = records[0].graph
        if set(graph.nodes) == {"node"} and len(graph.edges) == 1:
            node_hops = None
            sampled_node_ids = self._sample_node_ids(graph, records, return_hops=return_hops)
            if return_hops:
                node_ids, node_hops = sampled_node_ids
            else:
                node_ids = sampled_node_ids
            subgraph, node_mapping = self._subgraph(graph, node_ids)
            sampled = [self._local_record(record, subgraph, node_mapping) for record in records]
            sampled_payload = sampled if is_sequence else sampled[0]
            if return_hops:
                return sampled_payload, {"link_node_ids_local": node_ids, "link_node_hops": node_hops}
            return sampled_payload
        node_hops_by_type = None
        node_ids_by_type = self._hetero_sample_node_ids(graph, records, return_hops=return_hops)
        if return_hops:
            node_ids_by_type, node_hops_by_type = node_ids_by_type
        subgraph, node_mapping = self._hetero_subgraph(graph, node_ids_by_type)
        sampled = [self._hetero_local_record(record, subgraph, node_mapping) for record in records]
        sampled_payload = sampled if is_sequence else sampled[0]
        if return_hops:
            return sampled_payload, {
                "link_node_hops_by_type": node_hops_by_type,
                "link_block_edge_type": _single_link_edge_type(records, context="LinkNeighborSampler"),
            }
        if is_sequence:
            return sampled
        return sampled[0]

    def build_plan(self, item) -> SamplingPlan | tuple[()]:
        records, is_sequence = self._seed_records(item)
        if not records:
            return ()
        graph = records[0].graph
        if self.output_blocks:
            _single_link_edge_type(records, context="LinkNeighborSampler")
        plan = SamplingPlan(
            request=LinkSeedRequest(
                src_ids=torch.tensor([_as_python_int(record.src_index) for record in records], dtype=torch.long),
                dst_ids=torch.tensor([_as_python_int(record.dst_index) for record in records], dtype=torch.long),
                edge_type=_resolve_link_edge_type(records[0]) if len(records) == 1 else None,
                labels=torch.tensor([_as_python_int(record.label) for record in records], dtype=torch.long),
                metadata=dict(records[0].metadata),
            ),
            stages=(
                PlanStage(
                    "sample_link_neighbors",
                    params={
                        "sampler": self,
                        "records": tuple(records),
                        "is_sequence": is_sequence,
                        "output_blocks": self.output_blocks,
                    },
                ),
            ),
            graph=graph,
        )

        additional_stages = []
        node_index_key = "node_ids" if set(graph.nodes) == {"node"} and len(graph.edges) == 1 else "node_ids_by_type"
        edge_index_key = "edge_ids" if set(graph.nodes) == {"node"} and len(graph.edges) == 1 else "edge_ids_by_type"
        for node_type, feature_names in self._resolved_node_feature_names(graph):
            additional_stages.append(
                PlanStage(
                    "fetch_node_features",
                    params={
                        "node_type": node_type,
                        "feature_names": feature_names,
                        "index_key": node_index_key,
                        "output_key": f"node_features:{node_type}",
                    },
                )
            )
        for edge_type, feature_names in self._resolved_edge_feature_names(graph):
            additional_stages.append(
                PlanStage(
                    "fetch_edge_features",
                    params={
                        "edge_type": edge_type,
                        "feature_names": feature_names,
                        "index_key": edge_index_key,
                        "output_key": f"edge_features:{edge_type}",
                    },
                )
            )
        if additional_stages:
            return plan.append(*additional_stages)
        return plan

    def sample(self, item):
        plan = self.build_plan(item)
        if not isinstance(plan, SamplingPlan):
            return plan
        context = PlanExecutor().execute(plan, graph=plan.graph)
        return materialize_context(context)


class TemporalNeighborSampler(LinkNeighborSampler):
    def __init__(
        self,
        num_neighbors,
        *,
        time_window=None,
        max_events=None,
        strict_history=True,
        include_seed_timestamp: bool | None = None,
        seed=None,
        node_feature_names=None,
        edge_feature_names=None,
    ):
        if include_seed_timestamp is not None:
            strict_history = not bool(include_seed_timestamp)
        super().__init__(
            num_neighbors=num_neighbors,
            base_sampler=None,
            seed=seed,
            node_feature_names=node_feature_names,
            edge_feature_names=edge_feature_names,
        )
        resolved_time_window = None if time_window is None else _as_python_int(time_window)
        resolved_max_events = None if max_events is None else _as_python_int(max_events)
        if resolved_time_window is not None and resolved_time_window < 0:
            raise ValueError("time_window must be >= 0")
        if resolved_max_events is not None and resolved_max_events < 1:
            raise ValueError("max_events must be >= 1")
        self.time_window = resolved_time_window
        self.max_events = resolved_max_events
        self.strict_history = bool(strict_history)

    def _history_edge_ids(self, graph, edge_type, timestamp):
        if graph.schema.time_attr is None:
            raise ValueError("TemporalNeighborSampler requires a temporal graph with schema.time_attr")
        if edge_type not in graph.edges:
            raise ValueError("TemporalNeighborSampler record edge_type must exist in the source graph")

        edge_store = graph.edges[edge_type]
        sorted_timestamps, sorted_edge_ids, edge_ids_are_sorted = _temporal_history_lookup(
            edge_store,
            time_attr=graph.schema.time_attr,
        )
        query_time = torch.as_tensor(timestamp, dtype=sorted_timestamps.dtype, device=sorted_timestamps.device).view(1)
        if self.strict_history:
            upper = torch.searchsorted(sorted_timestamps, query_time, right=False)
        else:
            upper = torch.searchsorted(sorted_timestamps, query_time, right=True)
        lower = torch.zeros_like(upper)
        if self.time_window is not None:
            lower_bound = query_time - self.time_window
            lower = torch.searchsorted(sorted_timestamps, lower_bound, right=False)

        candidate_edge_ids = sorted_edge_ids[lower[0] : upper[0]]
        if self.max_events is not None and candidate_edge_ids.numel() > self.max_events:
            return candidate_edge_ids[-self.max_events :]
        if edge_ids_are_sorted or candidate_edge_ids.numel() <= 1:
            return candidate_edge_ids
        return torch.sort(candidate_edge_ids, stable=True).values

    def _next_frontier_from_edge_index(self, edge_index, frontier, visited, fanout):
        frontier_tensor = _sample_homo_next_frontier(
            edge_index,
            torch.tensor(sorted(frontier), dtype=torch.long, device=edge_index.device),
            torch.tensor(sorted(visited), dtype=torch.long, device=edge_index.device),
            fanout,
            generator=self._generator,
        )
        return {_as_python_int(node) for node in frontier_tensor}

    def _sample_node_ids(self, edge_index, src_index, dst_index):
        return _sample_homo_node_ids(
            edge_index,
            torch.tensor(
                [_as_python_int(src_index), _as_python_int(dst_index)],
                dtype=torch.long,
                device=edge_index.device,
            ),
            self.num_neighbors,
            generator=self._generator,
        )

    def _sample_history_node_ids(self, edge_store, history_edge_ids, src_index, dst_index):
        return _sample_homo_history_node_ids(
            edge_store,
            history_edge_ids,
            torch.tensor(
                [_as_python_int(src_index), _as_python_int(dst_index)],
                dtype=torch.long,
                device=edge_store.edge_index.device,
            ),
            self.num_neighbors,
            generator=self._generator,
        )

    def _relation_next_frontier(self, edge_index, frontier, visited, fanout, *, src_type, dst_type):
        candidates = _relation_neighbor_candidates(
            edge_index,
            frontier,
            visited,
            src_type=src_type,
            dst_type=dst_type,
        )
        next_frontier = {}
        for node_type, node_ids in candidates.items():
            sampled = _sample_fanout(node_ids, fanout, generator=self._generator)
            if sampled.numel() > 0:
                next_frontier[node_type] = sampled
        return next_frontier

    def _relation_history_next_frontier(self, edge_store, history_edge_ids, frontier, visited, fanout, *, src_type, dst_type):
        candidates = _history_relation_neighbor_candidates(
            edge_store,
            history_edge_ids,
            frontier,
            visited,
            src_type=src_type,
            dst_type=dst_type,
        )
        next_frontier = {}
        for node_type, node_ids in candidates.items():
            sampled = _sample_fanout(node_ids, fanout, generator=self._generator)
            if sampled.numel() > 0:
                next_frontier[node_type] = sampled
        return next_frontier

    def _relation_sample_node_ids(self, edge_index, src_index, dst_index, *, src_type, dst_type):
        unique_types = tuple(dict.fromkeys((src_type, dst_type)))
        visited = {
            node_type: torch.empty(0, dtype=torch.long, device=edge_index.device)
            for node_type in unique_types
        }
        visited[src_type] = _merge_sorted_unique_tensors(
            visited[src_type],
            torch.tensor([_as_python_int(src_index)], dtype=torch.long, device=edge_index.device),
        )
        visited[dst_type] = _merge_sorted_unique_tensors(
            visited[dst_type],
            torch.tensor([_as_python_int(dst_index)], dtype=torch.long, device=edge_index.device),
        )
        frontier = {node_type: node_ids.clone() for node_type, node_ids in visited.items()}
        for fanout in self.num_neighbors:
            frontier = self._relation_next_frontier(
                edge_index,
                frontier,
                visited,
                fanout,
                src_type=src_type,
                dst_type=dst_type,
            )
            for node_type, node_ids in frontier.items():
                visited[node_type] = _merge_sorted_unique_tensors(visited[node_type], node_ids)
            if not frontier:
                break
        return {node_type: node_ids.clone() for node_type, node_ids in visited.items()}

    def _relation_sample_history_node_ids(self, edge_store, history_edge_ids, src_index, dst_index, *, src_type, dst_type):
        edge_index = edge_store.edge_index
        unique_types = tuple(dict.fromkeys((src_type, dst_type)))
        visited = {
            node_type: torch.empty(0, dtype=torch.long, device=edge_index.device)
            for node_type in unique_types
        }
        visited[src_type] = _merge_sorted_unique_tensors(
            visited[src_type],
            torch.tensor([_as_python_int(src_index)], dtype=torch.long, device=edge_index.device),
        )
        visited[dst_type] = _merge_sorted_unique_tensors(
            visited[dst_type],
            torch.tensor([_as_python_int(dst_index)], dtype=torch.long, device=edge_index.device),
        )
        frontier = {node_type: node_ids.clone() for node_type, node_ids in visited.items()}
        for fanout in self.num_neighbors:
            frontier = self._relation_history_next_frontier(
                edge_store,
                history_edge_ids,
                frontier,
                visited,
                fanout,
                src_type=src_type,
                dst_type=dst_type,
            )
            for node_type, node_ids in frontier.items():
                visited[node_type] = _merge_sorted_unique_tensors(visited[node_type], node_ids)
            if not frontier:
                break
        return {node_type: node_ids.clone() for node_type, node_ids in visited.items()}

    def _subgraph(self, graph, edge_type, node_ids, history_edge_ids):
        node_ids = node_ids.to(dtype=torch.long)
        num_nodes = int(graph.x.size(0))
        edge_store = graph.edges[edge_type]
        edge_index = edge_store.edge_index

        history_edge_ids = torch.as_tensor(history_edge_ids, dtype=torch.long, device=edge_index.device)
        kept_edge_ids = _history_edge_ids_for_nodes(edge_store, history_edge_ids, node_ids, node_ids)

        if kept_edge_ids.numel() == 0:
            subgraph_edge_index = edge_index[:, :0]
        else:
            subgraph_edge_index = _relabel_bipartite_edge_index(
                edge_index[:, kept_edge_ids],
                node_ids,
                node_ids,
            )

        node_data = {}
        for key, value in graph.nodes["node"].data.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == num_nodes:
                node_data[key] = value[node_ids]
            else:
                node_data[key] = value
        if "n_id" not in node_data:
            node_data["n_id"] = node_ids

        edge_count = int(edge_index.size(1))
        edge_data = {"edge_index": subgraph_edge_index}
        for key, value in edge_store.data.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_data[key] = value[kept_edge_ids]
            else:
                edge_data[key] = value
        if "e_id" not in edge_data:
            edge_data["e_id"] = kept_edge_ids

        return (
            Graph.temporal(
                nodes={"node": node_data},
                edges={edge_type: edge_data},
                time_attr=graph.schema.time_attr,
            ),
            node_ids,
        )

    def _relation_subgraph(self, graph, edge_type, node_ids_by_type, history_edge_ids):
        src_type, _, dst_type = edge_type
        unique_node_types = tuple(dict.fromkeys((src_type, dst_type)))
        nodes = {}
        for node_type in unique_node_types:
            store = graph.nodes[node_type]
            node_ids = node_ids_by_type[node_type].to(dtype=torch.long)
            num_nodes = store.x.size(0)

            node_data = {}
            for key, value in store.data.items():
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == num_nodes:
                    node_data[key] = value[node_ids]
                else:
                    node_data[key] = value
            if "n_id" not in node_data:
                node_data["n_id"] = node_ids
            nodes[node_type] = node_data

        edge_store = graph.edges[edge_type]
        history_edge_ids = torch.as_tensor(history_edge_ids, dtype=torch.long, device=edge_store.edge_index.device)
        kept_edge_ids = _history_edge_ids_for_nodes(
            edge_store,
            history_edge_ids,
            node_ids_by_type[src_type],
            node_ids_by_type[dst_type],
        )
        if kept_edge_ids.numel() == 0:
            subgraph_edge_index = edge_store.edge_index[:, :0]
        else:
            subgraph_edge_index = _relabel_bipartite_edge_index(
                edge_store.edge_index[:, kept_edge_ids],
                node_ids_by_type[src_type],
                node_ids_by_type[dst_type],
            )

        edge_count = int(edge_store.edge_index.size(1))
        edge_data = {"edge_index": subgraph_edge_index}
        for key, value in edge_store.data.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_data[key] = value[kept_edge_ids]
            else:
                edge_data[key] = value
        if "e_id" not in edge_data:
            edge_data["e_id"] = kept_edge_ids

        return Graph.temporal(nodes=nodes, edges={edge_type: edge_data}, time_attr=graph.schema.time_attr), {
            node_type: node_ids_by_type[node_type].to(dtype=torch.long)
            for node_type in unique_node_types
        }

    def _local_record(self, record, graph, node_mapping, *, edge_type):
        local_positions = _lookup_positions(
            node_mapping,
            torch.tensor(
                [_as_python_int(record.src_index), _as_python_int(record.dst_index)],
                dtype=torch.long,
                device=node_mapping.device,
            ),
            entity_name="node",
        )
        return TemporalEventRecord(
            graph=graph,
            src_index=_as_python_int(local_positions[0]),
            dst_index=_as_python_int(local_positions[1]),
            timestamp=_as_python_int(record.timestamp),
            label=_as_python_int(record.label),
            event_features=record.event_features,
            metadata=dict(record.metadata),
            sample_id=record.sample_id,
            edge_type=edge_type,
        )

    def _hetero_local_record(self, record, graph, node_mapping, *, edge_type):
        _, src_type, dst_type = _temporal_endpoint_types(record)
        src_position = _lookup_positions(
            node_mapping[src_type],
            torch.tensor(
                [_as_python_int(record.src_index)],
                dtype=torch.long,
                device=node_mapping[src_type].device,
            ),
            entity_name="source node",
        )
        dst_position = _lookup_positions(
            node_mapping[dst_type],
            torch.tensor(
                [_as_python_int(record.dst_index)],
                dtype=torch.long,
                device=node_mapping[dst_type].device,
            ),
            entity_name="destination node",
        )
        return TemporalEventRecord(
            graph=graph,
            src_index=_as_python_int(src_position[0]),
            dst_index=_as_python_int(dst_position[0]),
            timestamp=_as_python_int(record.timestamp),
            label=_as_python_int(record.label),
            event_features=record.event_features,
            metadata=dict(record.metadata),
            sample_id=record.sample_id,
            edge_type=edge_type,
        )

    def _resolved_temporal_node_feature_names(self, graph, edge_type):
        src_type, _, dst_type = edge_type
        allowed_types = set((src_type, dst_type))
        return tuple(
            (node_type, feature_names)
            for node_type, feature_names in self._resolved_node_feature_names(graph)
            if node_type in allowed_types
        )

    def _resolved_temporal_edge_feature_names(self, graph, edge_type):
        return tuple(
            (current_edge_type, feature_names)
            for current_edge_type, feature_names in self._resolved_edge_feature_names(graph)
            if tuple(current_edge_type) == tuple(edge_type)
        )

    def _sample_event(self, item):
        if not isinstance(item, TemporalEventRecord):
            raise TypeError("TemporalNeighborSampler requires TemporalEventRecord items")
        edge_type = _resolve_temporal_edge_type(item)
        history_edge_ids = self._history_edge_ids(item.graph, edge_type, _as_python_int(item.timestamp))
        if set(item.graph.nodes) == {"node"} and len(item.graph.edges) == 1:
            node_ids = self._sample_history_node_ids(
                item.graph.edges[edge_type],
                history_edge_ids,
                item.src_index,
                item.dst_index,
            )
            subgraph, node_mapping = self._subgraph(item.graph, edge_type, node_ids, history_edge_ids)
            return self._local_record(item, subgraph, node_mapping, edge_type=edge_type)

        _, src_type, dst_type = _temporal_endpoint_types(item)
        node_ids_by_type = self._relation_sample_history_node_ids(
            item.graph.edges[edge_type],
            history_edge_ids,
            item.src_index,
            item.dst_index,
            src_type=src_type,
            dst_type=dst_type,
        )
        subgraph, node_mapping = self._relation_subgraph(item.graph, edge_type, node_ids_by_type, history_edge_ids)
        return self._hetero_local_record(item, subgraph, node_mapping, edge_type=edge_type)

    def build_plan(self, item) -> SamplingPlan:
        if not isinstance(item, TemporalEventRecord):
            raise TypeError("TemporalNeighborSampler requires TemporalEventRecord items")
        edge_type = _resolve_temporal_edge_type(item)
        plan_metadata = {}
        if item.sample_id is not None:
            plan_metadata["sample_id"] = item.sample_id
        graph = item.graph
        plan = SamplingPlan(
            request=TemporalSeedRequest(
                src_ids=torch.tensor([_as_python_int(item.src_index)], dtype=torch.long),
                dst_ids=torch.tensor([_as_python_int(item.dst_index)], dtype=torch.long),
                timestamps=torch.tensor([_as_python_int(item.timestamp)], dtype=torch.long),
                edge_type=edge_type,
                metadata=dict(item.metadata),
            ),
            stages=(
                PlanStage(
                    "sample_temporal_neighbors",
                    params={"sampler": self, "record": item},
                ),
            ),
            metadata=plan_metadata,
            graph=graph,
        )

        additional_stages = []
        node_index_key = "node_ids" if edge_type[0] == edge_type[2] == "node" else "node_ids_by_type"
        edge_index_key = "edge_ids" if edge_type[0] == edge_type[2] == "node" else "edge_ids_by_type"
        for node_type, feature_names in self._resolved_temporal_node_feature_names(graph, edge_type):
            additional_stages.append(
                PlanStage(
                    "fetch_node_features",
                    params={
                        "node_type": node_type,
                        "feature_names": feature_names,
                        "index_key": node_index_key,
                        "output_key": f"node_features:{node_type}",
                    },
                )
            )
        for current_edge_type, feature_names in self._resolved_temporal_edge_feature_names(graph, edge_type):
            additional_stages.append(
                PlanStage(
                    "fetch_edge_features",
                    params={
                        "edge_type": current_edge_type,
                        "feature_names": feature_names,
                        "index_key": edge_index_key,
                        "output_key": f"edge_features:{current_edge_type}",
                    },
                )
            )
        if additional_stages:
            return plan.append(*additional_stages)
        return plan

    def sample(self, item):
        plan = self.build_plan(item)
        context = PlanExecutor().execute(plan, graph=plan.graph)
        return materialize_context(context)


class NodeNeighborSampler(LinkNeighborSampler):
    def __init__(
        self,
        num_neighbors,
        *,
        seed=None,
        node_feature_names=None,
        edge_feature_names=None,
        output_blocks: bool = False,
        replace: bool = False,
        directed: bool = False,
    ):
        super().__init__(
            num_neighbors=num_neighbors,
            base_sampler=None,
            seed=seed,
            replace=replace,
            directed=directed,
        )
        self.node_feature_names = node_feature_names
        self.edge_feature_names = edge_feature_names
        self.output_blocks = bool(output_blocks)

    @staticmethod
    def _normalize_feature_names(feature_names):
        return tuple(str(name) for name in feature_names)

    def _resolved_node_feature_names(self, graph):
        if self.node_feature_names is None:
            return ()
        if isinstance(self.node_feature_names, dict):
            resolved = []
            for node_type, feature_names in self.node_feature_names.items():
                node_type = str(node_type)
                if node_type not in graph.nodes:
                    raise ValueError(f"unknown node_type for feature prefetch: {node_type!r}")
                names = self._normalize_feature_names(feature_names)
                if names:
                    resolved.append((node_type, names))
            return tuple(resolved)
        if set(graph.nodes) != {"node"} or len(graph.edges) != 1:
            raise ValueError("heterogeneous NodeNeighborSampler requires node_feature_names keyed by node type")
        names = self._normalize_feature_names(self.node_feature_names)
        if not names:
            return ()
        return (("node", names),)

    def _resolved_edge_feature_names(self, graph):
        if self.edge_feature_names is None:
            return ()
        if isinstance(self.edge_feature_names, dict):
            resolved = []
            for edge_type, feature_names in self.edge_feature_names.items():
                edge_type = tuple(edge_type)
                if edge_type not in graph.edges:
                    raise ValueError(f"unknown edge_type for feature prefetch: {edge_type!r}")
                names = self._normalize_feature_names(feature_names)
                if names:
                    resolved.append((edge_type, names))
            return tuple(resolved)
        if set(graph.nodes) != {"node"} or len(graph.edges) != 1:
            raise ValueError("heterogeneous NodeNeighborSampler requires edge_feature_names keyed by edge type")
        names = self._normalize_feature_names(self.edge_feature_names)
        if not names:
            return ()
        return ((graph._default_edge_type(), names),)

    @staticmethod
    def _normalize_seed_ids(seed) -> torch.Tensor:
        seed_ids = torch.as_tensor(seed, dtype=torch.long)
        if seed_ids.ndim == 0:
            seed_ids = seed_ids.view(1)
        elif seed_ids.ndim != 1:
            raise ValueError("NodeNeighborSampler seed metadata must be a scalar or rank-1 collection")
        if seed_ids.numel() == 0:
            raise ValueError("NodeNeighborSampler requires at least one seed")
        return seed_ids

    def _seed_item(self, item):
        if isinstance(item, SampleRecord):
            graph = item.graph
            metadata = dict(item.metadata)
            sample_id = item.sample_id
            source_graph_id = item.source_graph_id
            seed = item.subgraph_seed
            if seed is None:
                seed = metadata.get("seed")
        elif isinstance(item, tuple) and len(item) == 2:
            graph, raw_metadata = item
            metadata = dict(raw_metadata)
            sample_id = metadata.get("sample_id")
            source_graph_id = metadata.get("source_graph_id")
            seed = metadata.get("seed")
        else:
            raise TypeError("NodeNeighborSampler requires (graph, metadata) tuples or SampleRecord items")
        if seed is None:
            raise ValueError("NodeNeighborSampler requires metadata['seed'] or subgraph_seed")
        node_type = metadata.get("node_type")
        if set(graph.nodes) == {"node"} and len(graph.edges) == 1:
            node_type = "node"
        elif node_type is None:
            raise ValueError("NodeNeighborSampler requires metadata['node_type'] for heterogeneous graphs")
        elif node_type not in graph.nodes:
            raise ValueError("NodeNeighborSampler metadata['node_type'] must exist in the source graph")
        seed_ids = self._normalize_seed_ids(seed)
        num_nodes = int(graph.x.size(0)) if node_type == "node" and hasattr(graph, "x") else int(graph.nodes[node_type].x.size(0))
        if torch.any((seed_ids < 0) | (seed_ids >= num_nodes)):
            raise ValueError("NodeNeighborSampler seed must fall within the source graph node range")
        if node_type != "node":
            metadata.setdefault("node_type", node_type)
        return graph, metadata, sample_id, source_graph_id, seed_ids, node_type

    def build_plan(self, item) -> SamplingPlan:
        graph, metadata, sample_id, source_graph_id, seed_ids, node_type = self._seed_item(item)
        node_block_edge_type = None
        if self.output_blocks:
            node_block_edge_type = _single_inbound_node_block_edge_type(
                graph,
                node_type=node_type,
                context="NodeNeighborSampler",
            )
        plan_metadata = {}
        if sample_id is not None:
            plan_metadata["sample_id"] = sample_id
        if source_graph_id is not None:
            plan_metadata["source_graph_id"] = source_graph_id
        plan = SamplingPlan(
            request=NodeSeedRequest(
                node_ids=seed_ids.clone(),
                node_type=node_type,
                metadata=metadata,
            ),
            stages=(
                PlanStage(
                    "expand_neighbors",
                    params={
                        "num_neighbors": tuple(self.num_neighbors),
                        "output_blocks": self.output_blocks,
                        "node_block_edge_type": node_block_edge_type,
                    },
                ),
            ),
            metadata=plan_metadata,
            graph=graph,
        )

        additional_stages = []
        node_index_key = "node_ids" if set(graph.nodes) == {"node"} and len(graph.edges) == 1 else "node_ids_by_type"
        edge_index_key = "edge_ids" if set(graph.nodes) == {"node"} and len(graph.edges) == 1 else "edge_ids_by_type"
        for current_node_type, feature_names in self._resolved_node_feature_names(graph):
            additional_stages.append(
                PlanStage(
                    "fetch_node_features",
                    params={
                        "node_type": current_node_type,
                        "feature_names": feature_names,
                        "index_key": node_index_key,
                        "output_key": f"node_features:{current_node_type}",
                    },
                )
            )
        for edge_type, feature_names in self._resolved_edge_feature_names(graph):
            additional_stages.append(
                PlanStage(
                    "fetch_edge_features",
                    params={
                        "edge_type": edge_type,
                        "feature_names": feature_names,
                        "index_key": edge_index_key,
                        "output_key": f"edge_features:{edge_type}",
                    },
                )
            )
        if additional_stages:
            return plan.append(*additional_stages)
        return plan

    def sample(self, item):
        plan = self.build_plan(item)
        context = PlanExecutor().execute(plan, graph=plan.graph)
        return materialize_context(context)


class NodeSeedSubgraphSampler(Sampler):
    def sample(self, item):
        graph, metadata = item
        return SampleRecord(
            graph=graph,
            metadata=dict(metadata),
            sample_id=metadata.get("sample_id"),
            source_graph_id=metadata.get("source_graph_id"),
            subgraph_seed=metadata.get("seed"),
        )
