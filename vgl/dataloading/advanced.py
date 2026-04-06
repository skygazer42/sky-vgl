from __future__ import annotations

from dataclasses import dataclass

import torch

from vgl.dataloading.dataset import ListDataset
from vgl.dataloading.loader import Loader
from vgl.dataloading.records import SampleRecord
from vgl.dataloading.sampler import FullGraphSampler, Sampler
from vgl.graph.graph import Graph
from vgl.ops import khop_nodes, node_subgraph, random_walk
from vgl.ops.subgraph import _lookup_positions, _membership_mask, _positions_for_endpoint_values


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _require_homo_graph(graph: Graph, *, context: str) -> None:
    if set(graph.nodes) != {"node"} or len(graph.edges) != 1:
        raise ValueError(f"{context} currently supports homogeneous graphs only")


def _parse_graph_item(item, *, context: str):
    if isinstance(item, SampleRecord):
        graph = item.graph
        metadata = dict(item.metadata)
        sample_id = item.sample_id
        source_graph_id = item.source_graph_id
    elif isinstance(item, tuple) and len(item) == 2:
        graph, raw_metadata = item
        metadata = dict(raw_metadata)
        sample_id = metadata.get("sample_id")
        source_graph_id = metadata.get("source_graph_id")
    else:
        graph = item
        metadata = {}
        sample_id = None
        source_graph_id = None
    if not isinstance(graph, Graph):
        raise TypeError(f"{context} requires Graph inputs or (graph, metadata) tuples")
    return graph, metadata, sample_id, source_graph_id


def _ordered_unique(values) -> list[int]:
    unique = []
    seen = set()
    for value in values:
        value = int(value)
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


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


def _tensor_to_int_list(values) -> list[int]:
    tensor = torch.as_tensor(values, dtype=torch.long).view(-1)
    return [_as_python_int(value) for value in tensor]


def _tensor_rows_to_int_lists(values) -> list[list[int]]:
    tensor = torch.as_tensor(values, dtype=torch.long)
    if tensor.ndim == 1:
        return [_tensor_to_int_list(tensor)]
    return [_tensor_to_int_list(row) for row in tensor]


def _normalize_seed_values(value) -> list[int]:
    if isinstance(value, torch.Tensor):
        values = _tensor_to_int_list(value)
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if not values:
        raise ValueError("seed collections must contain at least one node id")
    return [_as_python_int(seed) for seed in values]


def _normalize_bounded_ids(value, *, upper_bound: int, field_name: str) -> list[int]:
    ids = _normalize_seed_values(value)
    for index in ids:
        if index < 0 or index >= upper_bound:
            raise ValueError(f"{field_name} must fall within the valid range")
    return ids


def _induced_edge_ids(graph: Graph, node_ids: list[int]) -> torch.Tensor:
    node_tensor = torch.as_tensor(node_ids, dtype=torch.long, device=graph.edge_index.device).view(-1)
    if node_tensor.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=graph.edge_index.device)
    node_tensor = _ordered_unique_tensor(node_tensor)
    edge_type = graph._default_edge_type()
    store = graph.edges[edge_type]
    candidate_edge_ids = _positions_for_endpoint_values(store, node_tensor, endpoint=0)
    if candidate_edge_ids.numel() == 0:
        return candidate_edge_ids
    keep = _membership_mask(graph.edge_index[1, candidate_edge_ids], node_tensor)
    return candidate_edge_ids[keep]


def _slice_homo_graph(graph: Graph, node_ids: list[int], edge_pairs: list[tuple[int, int]] | None = None) -> Graph:
    node_index = torch.as_tensor(node_ids, dtype=torch.long, device=graph.x.device).view(-1)
    node_data = {}
    num_nodes = int(graph.x.size(0))
    for key, value in graph.nodes["node"].data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.size(0)) == num_nodes:
            node_data[key] = value[node_index]
        else:
            node_data[key] = value
    node_data["n_id"] = node_index

    if edge_pairs is None:
        subgraph = node_subgraph(graph, node_index)
        edges = dict(subgraph.edges[subgraph._default_edge_type()].data)
        edges.setdefault("e_id", _induced_edge_ids(graph, node_index))
        return Graph.homo(edge_index=edges.pop("edge_index"), edge_data=edges, **node_data)

    if edge_pairs is not None and len(edge_pairs) > 0:
        edge_pairs_tensor = torch.as_tensor(edge_pairs, dtype=torch.long, device=graph.edge_index.device).view(-1, 2)
        lookup_index = node_index.to(device=graph.edge_index.device)
        src_pairs = edge_pairs_tensor[:, 0].contiguous()
        dst_pairs = edge_pairs_tensor[:, 1].contiguous()
        edge_index = torch.stack(
            (
                _lookup_positions(lookup_index, src_pairs, entity_name="source node"),
                _lookup_positions(lookup_index, dst_pairs, entity_name="destination node"),
            ),
            dim=0,
        ).to(dtype=torch.long, device=graph.edge_index.device)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=graph.edge_index.device)
    return Graph.homo(edge_index=edge_index, **node_data)


def _path_graph_from_walk(graph: Graph, walk: list[int]) -> Graph:
    return _path_graph_from_walks(graph, [walk])


def _path_graph_from_walks(graph: Graph, walks: list[list[int]]) -> Graph:
    walk_tensor = torch.as_tensor(walks, dtype=torch.long, device=graph.edge_index.device)
    if walk_tensor.ndim == 1:
        walk_tensor = walk_tensor.view(1, -1)
    valid_nodes = walk_tensor.reshape(-1)
    valid_nodes = valid_nodes[valid_nodes >= 0]
    node_ids = _ordered_unique_tensor(valid_nodes)
    if walk_tensor.size(1) > 1:
        src = walk_tensor[:, :-1].reshape(-1)
        dst = walk_tensor[:, 1:].reshape(-1)
        edge_mask = (src >= 0) & (dst >= 0)
        if bool(edge_mask.any()):
            edge_pairs = torch.stack((src[edge_mask], dst[edge_mask]), dim=1)
        else:
            edge_pairs = torch.empty((0, 2), dtype=torch.long, device=walk_tensor.device)
    else:
        edge_pairs = torch.empty((0, 2), dtype=torch.long, device=walk_tensor.device)
    if node_ids.numel() == 0:
        node_ids = torch.tensor([0], dtype=torch.long, device=walk_tensor.device)
    return _slice_homo_graph(graph, node_ids, edge_pairs=edge_pairs)


def _walk_lengths(walks: list[list[int]]) -> list[int]:
    return [sum(1 for node_id in walk if int(node_id) >= 0) for walk in walks]


def _walk_edge_pairs(walks: list[list[int]]) -> list[list[list[int]]]:
    walk_pairs = []
    for walk in walks:
        pairs = []
        for src, dst in zip(walk[:-1], walk[1:]):
            if int(src) < 0 or int(dst) < 0:
                continue
            pairs.append([int(src), int(dst)])
        walk_pairs.append(pairs)
    return walk_pairs


def _walk_ended_early(walks: list[list[int]], *, walk_length: int) -> list[bool]:
    expected_nodes = int(walk_length) + 1
    return [_walk_length < expected_nodes for _walk_length in _walk_lengths(walks)]


def _num_walks_ended_early(walks: list[list[int]], *, walk_length: int) -> int:
    return sum(1 for ended_early in _walk_ended_early(walks, walk_length=walk_length) if ended_early)


def _walk_start_positions(sampled_node_ids: list[int], walk_starts: list[int]) -> list[int]:
    local_positions = {int(node_id): index for index, node_id in enumerate(sampled_node_ids)}
    return [int(local_positions[int(node_id)]) for node_id in walk_starts]


def _seed_positions(sampled_node_ids: list[int], seed_ids: list[int]) -> list[int]:
    local_positions = {int(node_id): index for index, node_id in enumerate(sampled_node_ids)}
    return [int(local_positions[int(node_id)]) for node_id in seed_ids]


def _sampled_node_ids(graph: Graph) -> list[int]:
    return _tensor_to_int_list(graph.nodes["node"].data["n_id"])


def _sampled_edge_ids(graph: Graph) -> list[int] | None:
    edge_ids = graph.edges[graph._default_edge_type()].data.get("e_id")
    if edge_ids is None:
        return None
    return _tensor_to_int_list(edge_ids)


def _enrich_sample_metadata(metadata: dict, sample_graph: Graph) -> dict:
    payload = dict(metadata)
    payload.setdefault("sampled_node_ids", _sampled_node_ids(sample_graph))
    payload["sampled_num_nodes"] = int(sample_graph.x.size(0))
    payload["sampled_num_edges"] = int(sample_graph.edge_index.size(1))
    edge_ids = _sampled_edge_ids(sample_graph)
    if edge_ids is not None:
        payload.setdefault("sampled_edge_ids", edge_ids)
        payload["subgraph_edge_ids"] = edge_ids
    return payload


def _seeded_sample_records(
    sample_graph: Graph,
    *,
    metadata: dict,
    sample_id,
    source_graph_id,
    seed_ids: list[int],
    sampled_node_ids: list[int],
):
    local_seed_positions = {int(node_id): index for index, node_id in enumerate(sampled_node_ids)}
    samples = []
    for seed_id in seed_ids:
        sample_metadata = dict(metadata)
        sample_metadata["seed"] = int(seed_id)
        samples.append(
            SampleRecord(
                graph=sample_graph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                subgraph_seed=local_seed_positions[int(seed_id)],
            )
        )
    if len(samples) == 1:
        return samples[0]
    return samples


def _finalize_metadata(
    metadata: dict,
    *,
    sampler_name: str,
    sample_id,
    seed_ids=None,
    sampling_config: dict | None = None,
) -> dict:
    payload = dict(metadata)
    payload.setdefault("sample_id", sample_id)
    payload.setdefault("sampler", sampler_name)
    if seed_ids is not None:
        payload["seed_ids"] = [int(value) for value in seed_ids]
    payload["sampling_config"] = dict(sampling_config or {})
    return payload


class RandomWalkSampler(Sampler):
    def __init__(self, *, walk_length: int, num_walks: int = 1, seed: int | None = None, edge_type=None, expand_seeds: bool = False):
        resolved_walk_length = _as_python_int(walk_length)
        resolved_num_walks = _as_python_int(num_walks)
        if resolved_walk_length < 0:
            raise ValueError("walk_length must be >= 0")
        if resolved_num_walks < 1:
            raise ValueError("num_walks must be >= 1")
        self.walk_length = resolved_walk_length
        self.num_walks = resolved_num_walks
        self.seed = seed
        self.edge_type = None if edge_type is None else tuple(edge_type)
        self.expand_seeds = bool(expand_seeds)
        self._generator = None
        if seed is not None:
            self._generator = torch.Generator()
            self._generator.manual_seed(_as_python_int(seed))

    def _sample_start(self, graph: Graph) -> int:
        num_nodes = int(graph.x.size(0))
        if self._generator is None:
            return _as_python_int(torch.randint(num_nodes, (1,)))
        return _as_python_int(torch.randint(num_nodes, (1,), generator=self._generator))

    def _seed_ids_from_metadata(self, graph: Graph, metadata: dict) -> list[int]:
        seed = metadata.get("seed")
        if seed is None:
            seeds = [self._sample_start(graph) for _ in range(self.num_walks)]
        else:
            seeds = _normalize_seed_values(seed)
            if len(seeds) == 1 and self.num_walks > 1:
                seeds = seeds * self.num_walks
        num_nodes = int(graph.x.size(0))
        for seed_id in seeds:
            if seed_id < 0 or seed_id >= num_nodes:
                raise ValueError("seed must fall within the graph node range")
        return seeds

    def _sample_walks(self, graph: Graph, seed_ids: list[int]) -> list[list[int]]:
        traces = random_walk(graph, seed_ids, length=self.walk_length, edge_type=self.edge_type)
        return _tensor_rows_to_int_lists(traces)

    def sample(self, item):
        graph, metadata, sample_id, source_graph_id = _parse_graph_item(item, context=self.__class__.__name__)
        _require_homo_graph(graph, context=self.__class__.__name__)
        explicit_seed_ids = _normalize_seed_values(metadata["seed"]) if "seed" in metadata else None
        explicit_multi_seed = explicit_seed_ids is not None and len(explicit_seed_ids) > 1
        seed_ids = self._seed_ids_from_metadata(graph, metadata)
        walks = self._sample_walks(graph, seed_ids)
        sample_graph = _path_graph_from_walks(graph, walks)
        sampled_node_ids = _sampled_node_ids(sample_graph)
        sample_metadata = _finalize_metadata(
            metadata,
            sampler_name=self.__class__.__name__,
            sample_id=sample_id,
            seed_ids=seed_ids,
            sampling_config={"walk_length": self.walk_length, "num_walks": self.num_walks, "edge_type": self.edge_type},
        )
        if "seed" not in sample_metadata and len(seed_ids) == 1:
            sample_metadata["seed"] = seed_ids[0]
        if len(walks) == 1:
            sample_metadata["walk"] = walks[0]
        else:
            sample_metadata.pop("walk", None)
        sample_metadata["walk_starts"] = [int(node_id) for node_id in seed_ids]
        sample_metadata["walk_nodes"] = sampled_node_ids
        sample_metadata["walk_start_positions"] = _walk_start_positions(sampled_node_ids, sample_metadata["walk_starts"])
        sample_metadata["walks"] = walks
        sample_metadata["sampled_num_walks"] = len(walks)
        sample_metadata["walk_lengths"] = _walk_lengths(walks)
        sample_metadata["walk_ended_early"] = _walk_ended_early(walks, walk_length=self.walk_length)
        sample_metadata["num_walks_ended_early"] = _num_walks_ended_early(walks, walk_length=self.walk_length)
        sample_metadata["walk_edge_pairs"] = _walk_edge_pairs(walks)
        sample_metadata.setdefault("walk_length", self.walk_length)
        sample_metadata["sampled_node_ids"] = sampled_node_ids
        sample_metadata = _enrich_sample_metadata(sample_metadata, sample_graph)
        if explicit_multi_seed and self.expand_seeds:
            return _seeded_sample_records(
                sample_graph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                seed_ids=explicit_seed_ids,
                sampled_node_ids=sampled_node_ids,
            )
        return SampleRecord(
            graph=sample_graph,
            metadata=sample_metadata,
            sample_id=sample_id,
            source_graph_id=source_graph_id,
        )


class Node2VecWalkSampler(RandomWalkSampler):
    def __init__(
        self,
        *,
        walk_length: int,
        num_walks: int = 1,
        p: float = 1.0,
        q: float = 1.0,
        seed: int | None = None,
        edge_type=None,
        expand_seeds: bool = False,
    ):
        super().__init__(walk_length=walk_length, num_walks=num_walks, seed=seed, edge_type=edge_type, expand_seeds=expand_seeds)
        if p <= 0 or q <= 0:
            raise ValueError("p and q must be > 0")
        self.p = float(p)
        self.q = float(q)

    def _adjacency(self, graph: Graph) -> tuple[dict[int, list[int]], dict[int, set[int]]]:
        adjacency: dict[int, list[int]] = {}
        adjacency_set: dict[int, set[int]] = {}
        for src_index, dst_index in zip(graph.edge_index[0], graph.edge_index[1]):
            src_value = _as_python_int(src_index)
            dst_value = _as_python_int(dst_index)
            adjacency.setdefault(src_value, []).append(dst_value)
            adjacency_set.setdefault(src_value, set()).add(dst_value)
        return adjacency, adjacency_set

    def _weighted_choice(self, neighbors: list[int], weights: list[float]) -> int:
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        probs = weight_tensor / weight_tensor.sum()
        if self._generator is None:
            index = _as_python_int(torch.multinomial(probs, 1))
        else:
            index = _as_python_int(torch.multinomial(probs, 1, generator=self._generator))
        return int(neighbors[index])

    def sample(self, item):
        graph, metadata, sample_id, source_graph_id = _parse_graph_item(item, context=self.__class__.__name__)
        _require_homo_graph(graph, context=self.__class__.__name__)
        explicit_seed_ids = _normalize_seed_values(metadata["seed"]) if "seed" in metadata else None
        explicit_multi_seed = explicit_seed_ids is not None and len(explicit_seed_ids) > 1
        seed_ids = self._seed_ids_from_metadata(graph, metadata)
        adjacency, adjacency_set = self._adjacency(graph)

        walks = []
        for seed in seed_ids:
            walk = [seed]
            previous = None
            current = seed
            for _ in range(self.walk_length):
                neighbors = adjacency.get(current, [])
                if not neighbors:
                    walk.append(-1)
                    current = -1
                    previous = current
                    continue
                if previous is None or previous < 0:
                    if self._generator is None:
                        choice = int(neighbors[_as_python_int(torch.randint(len(neighbors), (1,)))])
                    else:
                        choice = int(
                            neighbors[_as_python_int(torch.randint(len(neighbors), (1,), generator=self._generator))]
                        )
                else:
                    weights = []
                    previous_neighbors = adjacency_set.get(previous, set())
                    for neighbor in neighbors:
                        if neighbor == previous:
                            weights.append(1.0 / self.p)
                        elif neighbor in previous_neighbors:
                            weights.append(1.0)
                        else:
                            weights.append(1.0 / self.q)
                    choice = self._weighted_choice(neighbors, weights)
                walk.append(choice)
                previous, current = current, choice
            walks.append(walk)

        sample_graph = _path_graph_from_walks(graph, walks)
        sampled_node_ids = _sampled_node_ids(sample_graph)
        sample_metadata = _finalize_metadata(
            metadata,
            sampler_name=self.__class__.__name__,
            sample_id=sample_id,
            seed_ids=seed_ids,
            sampling_config={
                "walk_length": self.walk_length,
                "num_walks": self.num_walks,
                "p": self.p,
                "q": self.q,
                "edge_type": self.edge_type,
            },
        )
        if "seed" not in sample_metadata and len(seed_ids) == 1:
            sample_metadata["seed"] = seed_ids[0]
        if len(walks) == 1:
            sample_metadata["walk"] = walks[0]
        else:
            sample_metadata.pop("walk", None)
        sample_metadata["walk_starts"] = [int(node_id) for node_id in seed_ids]
        sample_metadata["walk_nodes"] = sampled_node_ids
        sample_metadata["walk_start_positions"] = _walk_start_positions(sampled_node_ids, sample_metadata["walk_starts"])
        sample_metadata["walks"] = walks
        sample_metadata["sampled_num_walks"] = len(walks)
        sample_metadata["walk_lengths"] = _walk_lengths(walks)
        sample_metadata["walk_ended_early"] = _walk_ended_early(walks, walk_length=self.walk_length)
        sample_metadata["num_walks_ended_early"] = _num_walks_ended_early(walks, walk_length=self.walk_length)
        sample_metadata["walk_edge_pairs"] = _walk_edge_pairs(walks)
        sample_metadata["p"] = self.p
        sample_metadata["q"] = self.q
        sample_metadata["sampled_node_ids"] = sampled_node_ids
        sample_metadata = _enrich_sample_metadata(sample_metadata, sample_graph)
        if explicit_multi_seed and self.expand_seeds:
            return _seeded_sample_records(
                sample_graph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                seed_ids=explicit_seed_ids,
                sampled_node_ids=sampled_node_ids,
            )
        return SampleRecord(
            graph=sample_graph,
            metadata=sample_metadata,
            sample_id=sample_id,
            source_graph_id=source_graph_id,
        )


class GraphSAINTNodeSampler(Sampler):
    def __init__(self, *, num_sampled_nodes: int, seed: int | None = None):
        resolved_num_sampled_nodes = _as_python_int(num_sampled_nodes)
        if resolved_num_sampled_nodes < 1:
            raise ValueError("num_sampled_nodes must be >= 1")
        self.num_sampled_nodes = resolved_num_sampled_nodes
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(_as_python_int(seed))

    def sample(self, item):
        graph, metadata, sample_id, source_graph_id = _parse_graph_item(item, context=self.__class__.__name__)
        _require_homo_graph(graph, context=self.__class__.__name__)
        num_nodes = int(graph.x.size(0))
        count = min(self.num_sampled_nodes, num_nodes)
        selected = _tensor_to_int_list(torch.randperm(num_nodes, generator=self._generator)[:count])
        seed_ids = None
        if "seed" in metadata:
            seed_ids = _normalize_bounded_ids(metadata["seed"], upper_bound=num_nodes, field_name="seed")
            target_count = max(count, len(_ordered_unique(seed_ids)))
            selected = _ordered_unique(seed_ids + selected)[:target_count]
        else:
            selected = _ordered_unique(selected)
        sample_graph = _slice_homo_graph(graph, selected)
        sample_metadata = _finalize_metadata(
            metadata,
            sampler_name=self.__class__.__name__,
            sample_id=sample_id,
            seed_ids=seed_ids if seed_ids is not None else selected,
            sampling_config={"num_sampled_nodes": self.num_sampled_nodes},
        )
        sample_metadata["sampled_node_ids"] = [int(node_id) for node_id in selected]
        sample_metadata["seed_positions"] = _seed_positions(sample_metadata["sampled_node_ids"], sample_metadata["seed_ids"])
        sample_metadata = _enrich_sample_metadata(sample_metadata, sample_graph)
        if seed_ids is not None:
            return _seeded_sample_records(
                sample_graph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                seed_ids=seed_ids,
                sampled_node_ids=sample_metadata["sampled_node_ids"],
            )
        return SampleRecord(graph=sample_graph, metadata=sample_metadata, sample_id=sample_id, source_graph_id=source_graph_id)


class GraphSAINTEdgeSampler(Sampler):
    def __init__(self, *, num_sampled_edges: int, seed: int | None = None):
        resolved_num_sampled_edges = _as_python_int(num_sampled_edges)
        if resolved_num_sampled_edges < 1:
            raise ValueError("num_sampled_edges must be >= 1")
        self.num_sampled_edges = resolved_num_sampled_edges
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(_as_python_int(seed))

    def sample(self, item):
        graph, metadata, sample_id, source_graph_id = _parse_graph_item(item, context=self.__class__.__name__)
        _require_homo_graph(graph, context=self.__class__.__name__)
        edge_count = int(graph.edge_index.size(1))
        count = min(self.num_sampled_edges, edge_count)
        forced_edge_ids = []
        if "edge_id" in metadata or "edge_ids" in metadata:
            forced_edge_ids = _normalize_bounded_ids(
                metadata.get("edge_ids", metadata.get("edge_id")),
                upper_bound=edge_count,
                field_name="edge_id",
            )
        target_count = max(count, len(_ordered_unique(forced_edge_ids)))
        selected_edge_ids = []
        seen = set()
        for edge_id in forced_edge_ids:
            edge_id = int(edge_id)
            if edge_id in seen:
                continue
            seen.add(edge_id)
            selected_edge_ids.append(edge_id)
            if len(selected_edge_ids) >= target_count:
                break
        if len(selected_edge_ids) < target_count:
            for edge_id_tensor in torch.randperm(edge_count, generator=self._generator):
                edge_id = _as_python_int(edge_id_tensor)
                if edge_id in seen:
                    continue
                seen.add(edge_id)
                selected_edge_ids.append(edge_id)
                if len(selected_edge_ids) >= target_count:
                    break
        edge_ids = torch.tensor(selected_edge_ids, dtype=torch.long, device=graph.edge_index.device)
        endpoints = _tensor_to_int_list(_ordered_unique_tensor(graph.edge_index[:, edge_ids].reshape(-1)))
        sample_graph = _slice_homo_graph(graph, endpoints)
        sample_metadata = _finalize_metadata(
            metadata,
            sampler_name=self.__class__.__name__,
            sample_id=sample_id,
            seed_ids=endpoints,
            sampling_config={"num_sampled_edges": self.num_sampled_edges},
        )
        sample_metadata["sampled_node_ids"] = [int(node_id) for node_id in endpoints]
        sample_metadata["sampled_edge_ids"] = [int(edge_id) for edge_id in selected_edge_ids]
        sample_metadata["seed_positions"] = _seed_positions(sample_metadata["sampled_node_ids"], sample_metadata["seed_ids"])
        sample_metadata = _enrich_sample_metadata(sample_metadata, sample_graph)
        return SampleRecord(graph=sample_graph, metadata=sample_metadata, sample_id=sample_id, source_graph_id=source_graph_id)


class GraphSAINTRandomWalkSampler(Sampler):
    def __init__(self, *, num_walks: int, walk_length: int, seed: int | None = None):
        resolved_num_walks = _as_python_int(num_walks)
        resolved_walk_length = _as_python_int(walk_length)
        if resolved_num_walks < 1:
            raise ValueError("num_walks must be >= 1")
        if resolved_walk_length < 0:
            raise ValueError("walk_length must be >= 0")
        self.num_walks = resolved_num_walks
        self.walk_length = resolved_walk_length
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(_as_python_int(seed))

    def sample(self, item):
        graph, metadata, sample_id, source_graph_id = _parse_graph_item(item, context=self.__class__.__name__)
        _require_homo_graph(graph, context=self.__class__.__name__)
        num_nodes = int(graph.x.size(0))
        explicit_seed_ids = None
        if "seed" in metadata:
            explicit_seed_ids = _normalize_bounded_ids(metadata["seed"], upper_bound=num_nodes, field_name="seed")
            if len(explicit_seed_ids) == 1 and self.num_walks > 1:
                starts = explicit_seed_ids * self.num_walks
            else:
                starts = list(explicit_seed_ids)
        else:
            starts = _tensor_to_int_list(torch.randint(num_nodes, (self.num_walks,), generator=self._generator))
        walks = random_walk(graph, starts, length=self.walk_length)
        walk_nodes = _tensor_to_int_list(_ordered_unique_tensor(walks.reshape(-1)[walks.reshape(-1) >= 0]))
        sample_graph = _slice_homo_graph(graph, walk_nodes)
        sample_metadata = _finalize_metadata(
            metadata,
            sampler_name=self.__class__.__name__,
            sample_id=sample_id,
            seed_ids=explicit_seed_ids if explicit_seed_ids is not None else starts,
            sampling_config={"num_walks": self.num_walks, "walk_length": self.walk_length},
        )
        if "seed" not in sample_metadata and len(starts) == 1:
            sample_metadata["seed"] = starts[0]
        sample_metadata["walk_starts"] = [int(node_id) for node_id in starts]
        sample_metadata["walk_nodes"] = walk_nodes
        sample_metadata["sampled_node_ids"] = walk_nodes
        sample_metadata["walk_start_positions"] = _walk_start_positions(walk_nodes, sample_metadata["walk_starts"])
        sample_metadata["walks"] = _tensor_rows_to_int_lists(walks)
        sample_metadata["sampled_num_walks"] = len(sample_metadata["walks"])
        sample_metadata["walk_lengths"] = _walk_lengths(sample_metadata["walks"])
        sample_metadata["walk_ended_early"] = _walk_ended_early(sample_metadata["walks"], walk_length=self.walk_length)
        sample_metadata["num_walks_ended_early"] = _num_walks_ended_early(sample_metadata["walks"], walk_length=self.walk_length)
        sample_metadata["walk_edge_pairs"] = _walk_edge_pairs(sample_metadata["walks"])
        sample_metadata = _enrich_sample_metadata(sample_metadata, sample_graph)
        if explicit_seed_ids is not None:
            return _seeded_sample_records(
                sample_graph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                seed_ids=explicit_seed_ids,
                sampled_node_ids=sample_metadata["sampled_node_ids"],
            )
        return SampleRecord(graph=sample_graph, metadata=sample_metadata, sample_id=sample_id, source_graph_id=source_graph_id)


@dataclass(slots=True)
class ClusterData(ListDataset):
    graph: Graph
    num_parts: int
    seed: int | None = None

    def __post_init__(self):
        _require_homo_graph(self.graph, context=self.__class__.__name__)
        resolved_num_parts = _as_python_int(self.num_parts)
        if resolved_num_parts < 1:
            raise ValueError("num_parts must be >= 1")
        self.num_parts = resolved_num_parts
        generator = torch.Generator()
        if self.seed is not None:
            generator.manual_seed(_as_python_int(self.seed))
        permutation = torch.randperm(int(self.graph.x.size(0)), generator=generator)
        samples = []
        for cluster_id, part in enumerate(torch.chunk(permutation, self.num_parts)):
            if part.numel() == 0:
                continue
            node_ids = _tensor_to_int_list(part)
            subgraph = _slice_homo_graph(self.graph, node_ids)
            metadata = _finalize_metadata(
                {
                    "cluster_id": cluster_id,
                    "partition_id": cluster_id,
                    "num_parts": self.num_parts,
                    "node_ids": node_ids,
                    "sampled_node_ids": list(node_ids),
                },
                sampler_name=self.__class__.__name__,
                sample_id=f"cluster:{cluster_id}",
                seed_ids=node_ids,
                sampling_config={"num_parts": self.num_parts},
            )
            metadata = _enrich_sample_metadata(metadata, subgraph)
            samples.append(
                SampleRecord(
                    graph=subgraph,
                    metadata=metadata,
                    sample_id=f"cluster:{cluster_id}",
                )
            )
        self.samples = samples
        ListDataset.__init__(self, samples)


class ClusterLoader(Loader):
    def __init__(self, dataset: ClusterData, batch_size: int, **kwargs):
        super().__init__(dataset=dataset, sampler=FullGraphSampler(), batch_size=batch_size, **kwargs)


class ShaDowKHopSampler(Sampler):
    def __init__(self, *, num_hops: int, direction: str = "out", edge_type=None):
        resolved_num_hops = _as_python_int(num_hops)
        if resolved_num_hops < 0:
            raise ValueError("num_hops must be >= 0")
        self.num_hops = resolved_num_hops
        self.direction = direction
        self.edge_type = edge_type

    def sample(self, item):
        graph, metadata, sample_id, source_graph_id = _parse_graph_item(item, context=self.__class__.__name__)
        _require_homo_graph(graph, context=self.__class__.__name__)
        if "seed" not in metadata:
            raise ValueError("ShaDowKHopSampler requires metadata['seed']")
        seed_ids = _normalize_bounded_ids(metadata["seed"], upper_bound=int(graph.x.size(0)), field_name="seed")
        node_ids = khop_nodes(graph, seed_ids, num_hops=self.num_hops, direction=self.direction, edge_type=self.edge_type)
        node_ids_list = _tensor_to_int_list(node_ids)
        subgraph = _slice_homo_graph(graph, node_ids_list)
        sample_metadata = _finalize_metadata(
            metadata,
            sampler_name=self.__class__.__name__,
            sample_id=sample_id,
            seed_ids=seed_ids,
            sampling_config={"num_hops": self.num_hops, "direction": self.direction, "edge_type": self.edge_type},
        )
        sample_metadata["sampled_node_ids"] = list(node_ids_list)
        sample_metadata["seed_positions"] = _seed_positions(sample_metadata["sampled_node_ids"], sample_metadata["seed_ids"])
        sample_metadata = _enrich_sample_metadata(sample_metadata, subgraph)
        if len(seed_ids) > 1:
            return _seeded_sample_records(
                subgraph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                seed_ids=seed_ids,
                sampled_node_ids=node_ids_list,
            )
        return SampleRecord(
            graph=subgraph,
            metadata=sample_metadata,
            sample_id=sample_id,
            source_graph_id=source_graph_id,
            subgraph_seed=node_ids_list.index(seed_ids[0]),
        )


__all__ = [
    "ClusterData",
    "ClusterLoader",
    "GraphSAINTEdgeSampler",
    "GraphSAINTNodeSampler",
    "GraphSAINTRandomWalkSampler",
    "Node2VecWalkSampler",
    "RandomWalkSampler",
    "ShaDowKHopSampler",
]
