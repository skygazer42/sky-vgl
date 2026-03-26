import torch

from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.materialize import materialize_context
from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.records import LinkPredictionRecord
from vgl.dataloading.records import SampleRecord
from vgl.dataloading.records import TemporalEventRecord
from vgl.dataloading.requests import LinkSeedRequest, NodeSeedRequest, TemporalSeedRequest
from vgl.graph.graph import Graph


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


def _single_link_edge_type(records, *, context: str) -> tuple[str, str, str]:
    edge_types = tuple(dict.fromkeys(_resolve_link_edge_type(record) for record in records))
    if len(edge_types) != 1:
        raise ValueError(f"{context} output_blocks requires a single edge_type")
    return edge_types[0]


def _single_inbound_node_block_edge_type(graph: Graph, *, node_type: str, context: str) -> tuple[str, str, str]:
    if set(graph.nodes) == {"node"} and len(graph.edges) == 1:
        return graph._default_edge_type()

    inbound_edge_types = tuple(edge_type for edge_type in graph.edges if edge_type[2] == node_type)
    if len(inbound_edge_types) == 1:
        return inbound_edge_types[0]
    if not inbound_edge_types:
        raise ValueError(
            f"{context} output_blocks requires exactly one inbound edge_type for node_type {node_type!r}; found none"
        )
    formatted = ", ".join(str(edge_type) for edge_type in inbound_edge_types)
    raise ValueError(
        f"{context} output_blocks requires exactly one inbound edge_type for node_type {node_type!r}; "
        f"found {len(inbound_edge_types)}: {formatted}"
    )


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


class Sampler:
    def sample(self, item):
        raise NotImplementedError


class FullGraphSampler(Sampler):
    def sample(self, graph):
        return graph


class UniformNegativeLinkSampler(Sampler):
    def __init__(self, num_negatives=1, *, exclude_positive_edges=True, exclude_seed_edges=False):
        if num_negatives < 1:
            raise ValueError("num_negatives must be >= 1")
        self.num_negatives = int(num_negatives)
        self.exclude_positive_edges = bool(exclude_positive_edges)
        self.exclude_seed_edges = bool(exclude_seed_edges)

    def _candidate_destinations(self, item):
        edge_type, _, dst_type = _link_endpoint_types(item)
        graph = item.graph
        num_nodes = int(graph.nodes[dst_type].x.size(0))
        mask = torch.ones(num_nodes, dtype=torch.bool)
        mask[int(item.dst_index)] = False
        if self.exclude_positive_edges:
            edge_index = graph.edges[edge_type].edge_index
            positive_mask = edge_index[0] == int(item.src_index)
            if positive_mask.any():
                mask[edge_index[1, positive_mask].to(dtype=torch.long)] = False
        return torch.arange(num_nodes, dtype=torch.long)[mask]

    def _positive_seed_record(self, item):
        query_id = item.query_id
        if query_id is None:
            query_id = item.sample_id if item.sample_id is not None else id(item)
        return LinkPredictionRecord(
            graph=item.graph,
            src_index=int(item.src_index),
            dst_index=int(item.dst_index),
            label=1,
            metadata=dict(item.metadata),
            sample_id=item.sample_id,
            exclude_seed_edge=self.exclude_seed_edges,
            hard_negative_dst=item.hard_negative_dst,
            candidate_dst=item.candidate_dst,
            edge_type=_resolve_link_edge_type(item),
            reverse_edge_type=_resolve_link_reverse_edge_type(item),
            query_id=query_id,
        )

    def _negative_record(self, item, dst_index, offset, *, hard_negative=False, query_id=None):
        negative_metadata = dict(item.metadata)
        negative_metadata["negative_sampled"] = True
        if hard_negative:
            negative_metadata["hard_negative_sampled"] = True
        suffix = "hard-neg" if hard_negative else "neg"
        return LinkPredictionRecord(
            graph=item.graph,
            src_index=int(item.src_index),
            dst_index=int(dst_index),
            label=0,
            metadata=negative_metadata,
            sample_id=None if item.sample_id is None else f"{item.sample_id}:{suffix}:{offset}",
            exclude_seed_edge=False,
            edge_type=_resolve_link_edge_type(item),
            reverse_edge_type=_resolve_link_reverse_edge_type(item),
            query_id=query_id,
        )

    def _uniform_destinations(self, item, count, *, excluded_destinations=()):
        candidates = self._candidate_destinations(item)
        if excluded_destinations:
            excluded = torch.tensor(list(excluded_destinations), dtype=torch.long, device=candidates.device)
            keep_mask = ~torch.isin(candidates, excluded)
            filtered = candidates[keep_mask]
            if filtered.numel() > 0:
                candidates = filtered
        if candidates.numel() == 0:
            raise ValueError("UniformNegativeLinkSampler could not find a valid negative destination")
        indices = torch.randint(candidates.numel(), (count,))
        return [int(candidates[index].item()) for index in indices.tolist()]

    def sample(self, item):
        if not isinstance(item, LinkPredictionRecord):
            raise TypeError("UniformNegativeLinkSampler requires LinkPredictionRecord items")
        if int(item.label) != 1:
            raise ValueError("UniformNegativeLinkSampler requires positive seed records")
        positive = self._positive_seed_record(item)
        sampled = [positive]
        destinations = self._uniform_destinations(item, self.num_negatives)
        for offset, dst_index in enumerate(destinations):
            sampled.append(self._negative_record(item, dst_index, offset, query_id=positive.query_id))
        return sampled


class HardNegativeLinkSampler(UniformNegativeLinkSampler):
    def __init__(
        self,
        num_negatives=1,
        *,
        num_hard_negatives=1,
        exclude_positive_edges=True,
        exclude_seed_edges=False,
    ):
        super().__init__(
            num_negatives=num_negatives,
            exclude_positive_edges=exclude_positive_edges,
            exclude_seed_edges=exclude_seed_edges,
        )
        if num_hard_negatives < 0:
            raise ValueError("num_hard_negatives must be >= 0")
        self.num_hard_negatives = int(num_hard_negatives)

    def _hard_negative_candidates(self, item):
        raw_candidates = item.hard_negative_dst
        if raw_candidates is None:
            raw_candidates = item.metadata.get("hard_negative_dst")
        if raw_candidates is None:
            return []
        candidates = torch.as_tensor(raw_candidates, dtype=torch.long).view(-1)
        _, _, dst_type = _link_endpoint_types(item)
        num_nodes = int(item.graph.nodes[dst_type].x.size(0))
        if ((candidates < 0) | (candidates >= num_nodes)).any():
            raise ValueError("hard_negative_dst entries must fall within the source graph node range")
        valid = self._candidate_destinations(item)
        valid_set = set(valid.tolist())
        unique_candidates = []
        for candidate in candidates.tolist():
            if candidate in valid_set and candidate not in unique_candidates:
                unique_candidates.append(int(candidate))
        return unique_candidates

    def sample(self, item):
        if not isinstance(item, LinkPredictionRecord):
            raise TypeError("HardNegativeLinkSampler requires LinkPredictionRecord items")
        if int(item.label) != 1:
            raise ValueError("HardNegativeLinkSampler requires positive seed records")

        positive = self._positive_seed_record(item)
        sampled = [positive]
        hard_candidates = self._hard_negative_candidates(item)
        hard_count = min(self.num_negatives, self.num_hard_negatives, len(hard_candidates))
        selected_hard = []
        if hard_count > 0:
            permutation = torch.randperm(len(hard_candidates))[:hard_count].tolist()
            selected_hard = [hard_candidates[index] for index in permutation]
            for offset, dst_index in enumerate(selected_hard):
                sampled.append(
                    self._negative_record(
                        item,
                        dst_index,
                        offset,
                        hard_negative=True,
                        query_id=positive.query_id,
                    )
                )

        remaining = self.num_negatives - len(selected_hard)
        if remaining > 0:
            uniform_destinations = self._uniform_destinations(
                item,
                remaining,
                excluded_destinations=selected_hard,
            )
            start_offset = len(selected_hard)
            for offset, dst_index in enumerate(uniform_destinations, start=start_offset):
                sampled.append(self._negative_record(item, dst_index, offset, query_id=positive.query_id))
        return sampled


class CandidateLinkSampler(UniformNegativeLinkSampler):
    def __init__(self, *, filter_known_positive_edges=True, exclude_seed_edges=False):
        super().__init__(
            num_negatives=1,
            exclude_positive_edges=False,
            exclude_seed_edges=exclude_seed_edges,
        )
        self.filter_known_positive_edges = bool(filter_known_positive_edges)

    def _candidate_destinations(self, item):
        raw_candidates = item.candidate_dst
        if raw_candidates is None:
            raw_candidates = item.metadata.get("candidate_dst")
        _, _, dst_type = _link_endpoint_types(item)
        num_nodes = int(item.graph.nodes[dst_type].x.size(0))
        if raw_candidates is None:
            return list(range(num_nodes))
        candidates = torch.as_tensor(raw_candidates, dtype=torch.long).view(-1)
        if ((candidates < 0) | (candidates >= num_nodes)).any():
            raise ValueError("candidate_dst entries must fall within the source graph node range")
        unique_candidates = []
        for candidate in candidates.tolist():
            candidate = int(candidate)
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _known_positive_destinations(self, item):
        edge_type = _resolve_link_edge_type(item)
        edge_index = item.graph.edges[edge_type].edge_index
        positive_mask = edge_index[0] == int(item.src_index)
        if not positive_mask.any():
            return set()
        return {int(dst_index) for dst_index in edge_index[1, positive_mask].tolist()}

    def sample(self, item):
        if not isinstance(item, LinkPredictionRecord):
            raise TypeError("CandidateLinkSampler requires LinkPredictionRecord items")
        if int(item.label) != 1:
            raise ValueError("CandidateLinkSampler requires positive seed records")

        positive = self._positive_seed_record(item)
        candidates = self._candidate_destinations(item)
        ordered_destinations = [int(positive.dst_index)]
        ordered_destinations.extend(candidate for candidate in candidates if int(candidate) != int(positive.dst_index))
        known_positive_destinations = set()
        if self.filter_known_positive_edges:
            known_positive_destinations = self._known_positive_destinations(item)

        sampled = [positive]
        for offset, dst_index in enumerate(ordered_destinations[1:]):
            negative = self._negative_record(item, dst_index, offset, query_id=positive.query_id)
            if dst_index in known_positive_destinations:
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
    ):
        if isinstance(num_neighbors, int):
            num_neighbors = [num_neighbors]
        self.num_neighbors = [int(value) for value in num_neighbors]
        if not self.num_neighbors:
            raise ValueError("num_neighbors must contain at least one hop")
        if any(value < -1 or value == 0 for value in self.num_neighbors):
            raise ValueError("num_neighbors entries must be positive integers or -1")
        self.base_sampler = base_sampler
        self.node_feature_names = node_feature_names
        self.edge_feature_names = edge_feature_names
        self.output_blocks = bool(output_blocks)
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(int(seed))

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
        sampled = self.base_sampler.sample(item) if self.base_sampler is not None else item
        is_sequence = isinstance(sampled, (list, tuple))
        records = list(sampled) if is_sequence else [sampled]
        if not records or any(not isinstance(record, LinkPredictionRecord) for record in records):
            raise TypeError("LinkNeighborSampler requires LinkPredictionRecord items")
        graph = records[0].graph
        if any(record.graph is not graph for record in records):
            raise ValueError("LinkNeighborSampler requires records from a single source graph")
        return records, is_sequence

    def _next_frontier(self, graph, frontier, visited, fanout):
        if not frontier:
            return set()
        edge_index = graph.edge_index
        frontier_tensor = torch.tensor(sorted(frontier), dtype=torch.long, device=edge_index.device)
        incident_mask = torch.isin(edge_index[0], frontier_tensor) | torch.isin(edge_index[1], frontier_tensor)
        if not incident_mask.any():
            return set()
        candidate_nodes = torch.unique(edge_index[:, incident_mask]).tolist()
        candidate_nodes = [int(node) for node in candidate_nodes if int(node) not in visited]
        if fanout != -1 and len(candidate_nodes) > fanout:
            permutation = torch.randperm(len(candidate_nodes), generator=self._generator)[:fanout].tolist()
            candidate_nodes = [candidate_nodes[index] for index in permutation]
        return set(candidate_nodes)

    def _sample_node_ids(self, graph, records, *, return_hops: bool = False):
        visited = {
            int(node)
            for record in records
            for node in (record.src_index, record.dst_index)
        }
        frontier = set(visited)
        hop_nodes = None
        if return_hops:
            hop_nodes = [torch.tensor(sorted(visited), dtype=torch.long, device=graph.edge_index.device)]
        for fanout in self.num_neighbors:
            frontier = self._next_frontier(graph, frontier, visited, fanout)
            visited.update(frontier)
            if hop_nodes is not None:
                hop_nodes.append(torch.tensor(sorted(visited), dtype=torch.long, device=graph.edge_index.device))
            elif not frontier:
                break
        sampled = torch.tensor(sorted(visited), dtype=torch.long, device=graph.edge_index.device)
        if hop_nodes is not None:
            return sampled, hop_nodes
        return sampled

    def _hetero_next_frontier(self, graph, frontier, visited, fanout):
        candidates = {node_type: set() for node_type in graph.schema.node_types}
        for edge_type, store in graph.edges.items():
            src_type, _, dst_type = edge_type
            src_frontier = frontier.get(src_type, set())
            dst_frontier = frontier.get(dst_type, set())
            if not src_frontier and not dst_frontier:
                continue
            edge_index = store.edge_index
            incident_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
            if src_frontier:
                src_tensor = torch.tensor(sorted(src_frontier), dtype=torch.long, device=edge_index.device)
                incident_mask |= torch.isin(edge_index[0], src_tensor)
            if dst_frontier:
                dst_tensor = torch.tensor(sorted(dst_frontier), dtype=torch.long, device=edge_index.device)
                incident_mask |= torch.isin(edge_index[1], dst_tensor)
            if not incident_mask.any():
                continue
            candidates[src_type].update(
                int(node)
                for node in edge_index[0, incident_mask].tolist()
                if int(node) not in visited[src_type]
            )
            candidates[dst_type].update(
                int(node)
                for node in edge_index[1, incident_mask].tolist()
                if int(node) not in visited[dst_type]
            )

        next_frontier = {}
        for node_type, values in candidates.items():
            candidate_list = sorted(values)
            if fanout != -1 and len(candidate_list) > fanout:
                permutation = torch.randperm(len(candidate_list), generator=self._generator)[:fanout].tolist()
                candidate_list = [candidate_list[index] for index in permutation]
            if candidate_list:
                next_frontier[node_type] = set(candidate_list)
        return next_frontier

    def _hetero_sample_node_ids(self, graph, records, *, return_hops: bool = False):
        def _snapshot(visited_by_type):
            return {
                node_type: torch.tensor(
                    sorted(node_ids),
                    dtype=torch.long,
                    device=next(iter(graph.nodes[node_type].data.values())).device,
                )
                for node_type, node_ids in visited_by_type.items()
            }

        visited = {node_type: set() for node_type in graph.schema.node_types}
        for record in records:
            edge_type, src_type, dst_type = _link_endpoint_types(record)
            if edge_type not in graph.edges:
                raise ValueError("LinkNeighborSampler record edge_type must exist in the source graph")
            visited[src_type].add(int(record.src_index))
            visited[dst_type].add(int(record.dst_index))
        frontier = {
            node_type: set(node_ids)
            for node_type, node_ids in visited.items()
        }
        hop_nodes = [_snapshot(visited)] if return_hops else None
        for fanout in self.num_neighbors:
            frontier = self._hetero_next_frontier(graph, frontier, visited, fanout)
            for node_type, node_ids in frontier.items():
                visited[node_type].update(node_ids)
            if hop_nodes is not None:
                hop_nodes.append(_snapshot(visited))
            elif not any(frontier.values()):
                break
        sampled = {
            node_type: torch.tensor(
                sorted(node_ids),
                dtype=torch.long,
                device=next(iter(graph.nodes[node_type].data.values())).device,
            )
            for node_type, node_ids in visited.items()
        }
        if hop_nodes is not None:
            return sampled, hop_nodes
        return sampled

    def _subgraph(self, graph, node_ids):
        node_ids = node_ids.to(dtype=torch.long)
        num_nodes = int(graph.x.size(0))
        edge_index = graph.edge_index
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        node_mask[node_ids] = True
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]

        node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        node_mapping[node_ids] = torch.arange(node_ids.size(0), dtype=torch.long, device=edge_index.device)
        subgraph_edge_index = node_mapping[edge_index[:, edge_mask]]

        node_data = {}
        for key, value in graph.nodes["node"].data.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == num_nodes:
                node_data[key] = value[node_ids]
            else:
                node_data[key] = value
        if "n_id" not in node_data:
            node_data["n_id"] = node_ids

        edge_store = graph.edges[graph._default_edge_type()]
        edge_count = int(edge_store.edge_index.size(1))
        edge_data = {}
        edge_ids = torch.arange(edge_count, dtype=torch.long, device=edge_index.device)[edge_mask]
        for key, value in edge_store.data.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_data[key] = value[edge_mask]
            else:
                edge_data[key] = value
        if "e_id" not in edge_data:
            edge_data["e_id"] = edge_ids
        return Graph.homo(edge_index=subgraph_edge_index, edge_data=edge_data, **node_data), node_mapping

    def _hetero_subgraph(self, graph, node_ids_by_type):
        node_masks = {}
        node_mappings = {}
        nodes = {}
        for node_type, store in graph.nodes.items():
            node_ids = node_ids_by_type[node_type]
            num_nodes = store.x.size(0)
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=store.x.device)
            if node_ids.numel() > 0:
                node_mask[node_ids] = True
            node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=store.x.device)
            node_mapping[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long, device=store.x.device)
            node_masks[node_type] = node_mask
            node_mappings[node_type] = node_mapping

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
            edge_index = store.edge_index
            edge_mask = node_masks[src_type][edge_index[0]] & node_masks[dst_type][edge_index[1]]
            subgraph_edge_index = torch.stack(
                [
                    node_mappings[src_type][edge_index[0, edge_mask]],
                    node_mappings[dst_type][edge_index[1, edge_mask]],
                ],
                dim=0,
            )
            edge_count = int(edge_index.size(1))
            edge_data = {"edge_index": subgraph_edge_index}
            edge_ids = torch.arange(edge_count, dtype=torch.long, device=edge_index.device)[edge_mask]
            for key, value in store.data.items():
                if key == "edge_index":
                    continue
                if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                    edge_data[key] = value[edge_mask]
                else:
                    edge_data[key] = value
            if "e_id" not in edge_data:
                edge_data["e_id"] = edge_ids
            edges[edge_type] = edge_data
        return Graph.hetero(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr), node_mappings

    def _local_record(self, record, graph, node_mapping):
        return LinkPredictionRecord(
            graph=graph,
            src_index=int(node_mapping[int(record.src_index)].item()),
            dst_index=int(node_mapping[int(record.dst_index)].item()),
            label=int(record.label),
            metadata=dict(record.metadata),
            sample_id=record.sample_id,
            exclude_seed_edge=bool(record.exclude_seed_edge),
            hard_negative_dst=record.hard_negative_dst,
            candidate_dst=record.candidate_dst,
            edge_type=_resolve_link_edge_type(record),
            reverse_edge_type=_resolve_link_reverse_edge_type(record),
            query_id=record.query_id,
            filter_ranking=bool(record.filter_ranking),
        )

    def _hetero_local_record(self, record, graph, node_mapping):
        edge_type, src_type, dst_type = _link_endpoint_types(record)
        return LinkPredictionRecord(
            graph=graph,
            src_index=int(node_mapping[src_type][int(record.src_index)].item()),
            dst_index=int(node_mapping[dst_type][int(record.dst_index)].item()),
            label=int(record.label),
            metadata=dict(record.metadata),
            sample_id=record.sample_id,
            exclude_seed_edge=bool(record.exclude_seed_edge),
            hard_negative_dst=record.hard_negative_dst,
            candidate_dst=record.candidate_dst,
            edge_type=edge_type,
            reverse_edge_type=_resolve_link_reverse_edge_type(record),
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

    def build_plan(self, item) -> SamplingPlan:
        records, is_sequence = self._seed_records(item)
        graph = records[0].graph
        if self.output_blocks:
            _single_link_edge_type(records, context="LinkNeighborSampler")
        plan = SamplingPlan(
            request=LinkSeedRequest(
                src_ids=torch.tensor([int(record.src_index) for record in records], dtype=torch.long),
                dst_ids=torch.tensor([int(record.dst_index) for record in records], dtype=torch.long),
                edge_type=_resolve_link_edge_type(records[0]) if len(records) == 1 else None,
                labels=torch.tensor([int(record.label) for record in records], dtype=torch.long),
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
        seed=None,
        node_feature_names=None,
        edge_feature_names=None,
    ):
        super().__init__(
            num_neighbors=num_neighbors,
            base_sampler=None,
            seed=seed,
            node_feature_names=node_feature_names,
            edge_feature_names=edge_feature_names,
        )
        if time_window is not None and int(time_window) < 0:
            raise ValueError("time_window must be >= 0")
        if max_events is not None and int(max_events) < 1:
            raise ValueError("max_events must be >= 1")
        self.time_window = None if time_window is None else int(time_window)
        self.max_events = None if max_events is None else int(max_events)
        self.strict_history = bool(strict_history)

    def _history_edge_ids(self, graph, edge_type, timestamp):
        if graph.schema.time_attr is None:
            raise ValueError("TemporalNeighborSampler requires a temporal graph with schema.time_attr")
        if edge_type not in graph.edges:
            raise ValueError("TemporalNeighborSampler record edge_type must exist in the source graph")

        edge_store = graph.edges[edge_type]
        edge_timestamps = edge_store.data[graph.schema.time_attr]
        if self.strict_history:
            edge_mask = edge_timestamps < timestamp
        else:
            edge_mask = edge_timestamps <= timestamp
        if self.time_window is not None:
            edge_mask &= edge_timestamps >= (timestamp - self.time_window)

        edge_ids = torch.arange(edge_store.edge_index.size(1), dtype=torch.long, device=edge_store.edge_index.device)
        edge_ids = edge_ids[edge_mask]
        if self.max_events is not None and edge_ids.numel() > self.max_events:
            order = torch.argsort(edge_timestamps[edge_ids], stable=True)
            edge_ids = edge_ids[order][-self.max_events :]
        return edge_ids

    def _next_frontier_from_edge_index(self, edge_index, frontier, visited, fanout):
        if not frontier or edge_index.numel() == 0:
            return set()
        frontier_tensor = torch.tensor(sorted(frontier), dtype=torch.long, device=edge_index.device)
        incident_mask = torch.isin(edge_index[0], frontier_tensor) | torch.isin(edge_index[1], frontier_tensor)
        if not incident_mask.any():
            return set()
        candidate_nodes = torch.unique(edge_index[:, incident_mask]).tolist()
        candidate_nodes = [int(node) for node in candidate_nodes if int(node) not in visited]
        if fanout != -1 and len(candidate_nodes) > fanout:
            permutation = torch.randperm(len(candidate_nodes), generator=self._generator)[:fanout].tolist()
            candidate_nodes = [candidate_nodes[index] for index in permutation]
        return set(candidate_nodes)

    def _sample_node_ids(self, edge_index, src_index, dst_index):
        visited = {int(src_index), int(dst_index)}
        frontier = set(visited)
        for fanout in self.num_neighbors:
            frontier = self._next_frontier_from_edge_index(edge_index, frontier, visited, fanout)
            visited.update(frontier)
            if not frontier:
                break
        return torch.tensor(sorted(visited), dtype=torch.long, device=edge_index.device)

    def _relation_next_frontier(self, edge_index, frontier, visited, fanout, *, src_type, dst_type):
        if edge_index.numel() == 0:
            return {node_type: set() for node_type in dict.fromkeys((src_type, dst_type))}
        incident_mask = torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        src_frontier = frontier.get(src_type, set())
        dst_frontier = frontier.get(dst_type, set())
        if src_frontier:
            src_tensor = torch.tensor(sorted(src_frontier), dtype=torch.long, device=edge_index.device)
            incident_mask |= torch.isin(edge_index[0], src_tensor)
        if dst_frontier:
            dst_tensor = torch.tensor(sorted(dst_frontier), dtype=torch.long, device=edge_index.device)
            incident_mask |= torch.isin(edge_index[1], dst_tensor)
        if not incident_mask.any():
            return {node_type: set() for node_type in dict.fromkeys((src_type, dst_type))}

        candidates = {node_type: set() for node_type in dict.fromkeys((src_type, dst_type))}
        candidates[src_type].update(
            int(node)
            for node in edge_index[0, incident_mask].tolist()
            if int(node) not in visited[src_type]
        )
        candidates[dst_type].update(
            int(node)
            for node in edge_index[1, incident_mask].tolist()
            if int(node) not in visited[dst_type]
        )

        next_frontier = {}
        for node_type, values in candidates.items():
            candidate_list = sorted(values)
            if fanout != -1 and len(candidate_list) > fanout:
                permutation = torch.randperm(len(candidate_list), generator=self._generator)[:fanout].tolist()
                candidate_list = [candidate_list[index] for index in permutation]
            next_frontier[node_type] = set(candidate_list)
        return next_frontier

    def _relation_sample_node_ids(self, edge_index, src_index, dst_index, *, src_type, dst_type):
        visited = {node_type: set() for node_type in dict.fromkeys((src_type, dst_type))}
        visited[src_type].add(int(src_index))
        visited[dst_type].add(int(dst_index))
        frontier = {node_type: set(node_ids) for node_type, node_ids in visited.items()}
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
                visited[node_type].update(node_ids)
            if not any(frontier.values()):
                break
        device = edge_index.device
        return {
            node_type: torch.tensor(sorted(node_ids), dtype=torch.long, device=device)
            for node_type, node_ids in visited.items()
        }

    def _subgraph(self, graph, edge_type, node_ids, history_edge_ids):
        node_ids = node_ids.to(dtype=torch.long)
        num_nodes = int(graph.x.size(0))
        edge_store = graph.edges[edge_type]
        edge_index = edge_store.edge_index

        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        node_mask[node_ids] = True
        history_edge_index = edge_index[:, history_edge_ids]
        edge_mask = node_mask[history_edge_index[0]] & node_mask[history_edge_index[1]]
        kept_edge_ids = history_edge_ids[edge_mask]

        node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=edge_index.device)
        node_mapping[node_ids] = torch.arange(node_ids.size(0), dtype=torch.long, device=edge_index.device)
        subgraph_edge_index = node_mapping[history_edge_index[:, edge_mask]]

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
            node_mapping,
        )

    def _relation_subgraph(self, graph, edge_type, node_ids_by_type, history_edge_ids):
        src_type, _, dst_type = edge_type
        unique_node_types = tuple(dict.fromkeys((src_type, dst_type)))
        node_masks = {}
        node_mappings = {}
        nodes = {}
        for node_type in unique_node_types:
            store = graph.nodes[node_type]
            node_ids = node_ids_by_type[node_type].to(dtype=torch.long)
            num_nodes = store.x.size(0)
            node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=store.x.device)
            if node_ids.numel() > 0:
                node_mask[node_ids] = True
            node_mapping = torch.full((num_nodes,), -1, dtype=torch.long, device=store.x.device)
            node_mapping[node_ids] = torch.arange(node_ids.numel(), dtype=torch.long, device=store.x.device)
            node_masks[node_type] = node_mask
            node_mappings[node_type] = node_mapping

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
        history_edge_index = edge_store.edge_index[:, history_edge_ids]
        edge_mask = node_masks[src_type][history_edge_index[0]] & node_masks[dst_type][history_edge_index[1]]
        kept_edge_ids = history_edge_ids[edge_mask]
        subgraph_edge_index = torch.stack(
            [
                node_mappings[src_type][history_edge_index[0, edge_mask]],
                node_mappings[dst_type][history_edge_index[1, edge_mask]],
            ],
            dim=0,
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

        return Graph.temporal(nodes=nodes, edges={edge_type: edge_data}, time_attr=graph.schema.time_attr), node_mappings

    def _local_record(self, record, graph, node_mapping, *, edge_type):
        return TemporalEventRecord(
            graph=graph,
            src_index=int(node_mapping[int(record.src_index)].item()),
            dst_index=int(node_mapping[int(record.dst_index)].item()),
            timestamp=int(record.timestamp),
            label=int(record.label),
            event_features=record.event_features,
            metadata=dict(record.metadata),
            sample_id=record.sample_id,
            edge_type=edge_type,
        )

    def _hetero_local_record(self, record, graph, node_mapping, *, edge_type):
        _, src_type, dst_type = _temporal_endpoint_types(record)
        return TemporalEventRecord(
            graph=graph,
            src_index=int(node_mapping[src_type][int(record.src_index)].item()),
            dst_index=int(node_mapping[dst_type][int(record.dst_index)].item()),
            timestamp=int(record.timestamp),
            label=int(record.label),
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
        history_edge_ids = self._history_edge_ids(item.graph, edge_type, int(item.timestamp))
        if set(item.graph.nodes) == {"node"} and len(item.graph.edges) == 1:
            history_edge_index = item.graph.edges[edge_type].edge_index[:, history_edge_ids]
            node_ids = self._sample_node_ids(history_edge_index, item.src_index, item.dst_index)
            subgraph, node_mapping = self._subgraph(item.graph, edge_type, node_ids, history_edge_ids)
            return self._local_record(item, subgraph, node_mapping, edge_type=edge_type)

        _, src_type, dst_type = _temporal_endpoint_types(item)
        history_edge_index = item.graph.edges[edge_type].edge_index[:, history_edge_ids]
        node_ids_by_type = self._relation_sample_node_ids(
            history_edge_index,
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
                src_ids=torch.tensor([int(item.src_index)], dtype=torch.long),
                dst_ids=torch.tensor([int(item.dst_index)], dtype=torch.long),
                timestamps=torch.tensor([int(item.timestamp)], dtype=torch.long),
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
    ):
        super().__init__(num_neighbors=num_neighbors, base_sampler=None, seed=seed)
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
