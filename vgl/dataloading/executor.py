from dataclasses import dataclass, field

import torch
from typing import Any, Callable

from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.records import LinkPredictionRecord, SampleRecord, TemporalEventRecord
from vgl.graph.graph import Graph
from vgl.storage.base import TensorSlice

StageHandler = Callable[[PlanStage, "MaterializationContext"], "MaterializationContext"]


def _resolve_feature_store(feature_store, graph):
    if feature_store is not None:
        return feature_store
    return getattr(graph, "feature_store", None)


def _resolve_state_index(context: "MaterializationContext", index_key: str, *, type_name) -> Any:
    index = context.state[index_key]
    if isinstance(index, dict):
        try:
            return index[type_name]
        except KeyError as exc:
            raise KeyError(f"missing staged index for {type_name!r}") from exc
    return index


def _resolve_fetch_index(stage: "PlanStage", context: "MaterializationContext", *, type_name, routed_source: bool):
    if routed_source:
        global_index_key = f"{stage.params['index_key']}_global"
        if global_index_key in context.state:
            return _resolve_state_index(context, global_index_key, type_name=type_name)
    return _resolve_state_index(context, stage.params["index_key"], type_name=type_name)


def _resolve_link_record_edge_type(record) -> tuple[str, str, str]:
    edge_type = getattr(record, "edge_type", None)
    if edge_type is None:
        edge_type = getattr(record, "metadata", {}).get("edge_type")
    if edge_type is not None:
        return tuple(edge_type)
    graph = record.graph
    if len(graph.edges) == 1:
        return next(iter(graph.edges))
    try:
        return graph._default_edge_type()
    except AttributeError as exc:
        raise ValueError("Link prediction on heterogeneous graphs requires edge_type") from exc


def _single_link_record_edge_type(records) -> tuple[str, str, str] | None:
    edge_types = tuple(dict.fromkeys(_resolve_link_record_edge_type(record) for record in records))
    if len(edge_types) == 1:
        return edge_types[0]
    return None


def _resolve_temporal_record_edge_type(record) -> tuple[str, str, str]:
    edge_type = getattr(record, "edge_type", None)
    if edge_type is None:
        edge_type = getattr(record, "metadata", {}).get("edge_type")
    if edge_type is not None:
        return tuple(edge_type)
    graph = record.graph
    if len(graph.edges) == 1:
        return next(iter(graph.edges))
    try:
        return graph._default_edge_type()
    except AttributeError as exc:
        raise ValueError("Temporal event prediction on heterogeneous graphs requires edge_type") from exc


def _graph_node_global_ids(graph, node_ids: torch.Tensor, *, node_type: str) -> torch.Tensor:
    node_ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    node_store = graph.nodes[str(node_type)]
    global_ids = node_store.data.get("n_id")
    if global_ids is None:
        return node_ids
    return torch.as_tensor(global_ids, dtype=torch.long, device=node_ids.device)[node_ids]


def _graph_edge_global_ids(graph, edge_ids: torch.Tensor, *, edge_type) -> torch.Tensor:
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).view(-1)
    edge_store = graph.edges[tuple(edge_type)]
    global_ids = edge_store.data.get("e_id")
    if global_ids is None:
        return edge_ids
    return torch.as_tensor(global_ids, dtype=torch.long, device=edge_ids.device)[edge_ids]


def _induced_edge_ids(graph, node_ids: torch.Tensor) -> torch.Tensor:
    node_ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    edge_index = graph.edge_index
    node_mask = torch.zeros(graph.x.size(0), dtype=torch.bool, device=edge_index.device)
    if node_ids.numel() > 0:
        node_mask[node_ids] = True
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    return torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)[edge_mask]


def _induced_edge_ids_by_type(graph, node_ids_by_type: dict[str, torch.Tensor]) -> dict[tuple[str, str, str], torch.Tensor]:
    edge_ids_by_type = {}
    for edge_type, store in graph.edges.items():
        src_type, _, dst_type = edge_type
        edge_index = store.edge_index
        src_ids = torch.as_tensor(node_ids_by_type[src_type], dtype=torch.long).view(-1)
        dst_ids = torch.as_tensor(node_ids_by_type[dst_type], dtype=torch.long).view(-1)
        src_mask = torch.isin(edge_index[0], src_ids) if src_ids.numel() > 0 else torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        dst_mask = torch.isin(edge_index[1], dst_ids) if dst_ids.numel() > 0 else torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        edge_mask = src_mask & dst_mask
        edge_ids_by_type[edge_type] = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)[edge_mask]
    return edge_ids_by_type


def _record_materialized_features(context: "MaterializationContext", *, entity_kind: str, type_name, fetched: dict[str, Any]) -> None:
    state_key = "_materialized_node_features" if entity_kind == "node" else "_materialized_edge_features"
    materialized = context.state.setdefault(state_key, {})
    type_bucket = materialized.setdefault(type_name, {})
    type_bucket.update(fetched)


def _infer_data_device(data: dict[str, Any], *, fallback=None):
    for value in data.values():
        if isinstance(value, torch.Tensor):
            return value.device
    if fallback is not None:
        return fallback
    return torch.device("cpu")


def _store_sampled_graph_indices(context: "MaterializationContext", graph) -> None:
    if set(graph.nodes) == {"node"} and len(graph.edges) == 1:
        node_store = graph.nodes["node"]
        node_ids = node_store.data.get("n_id")
        if node_ids is None:
            node_ids = torch.arange(
                graph._node_count("node"),
                dtype=torch.long,
                device=_infer_data_device(node_store.data, fallback=graph.edge_index.device),
            )
        edge_type = graph._default_edge_type()
        edge_store = graph.edges[edge_type]
        edge_ids = edge_store.data.get("e_id")
        if edge_ids is None:
            edge_ids = torch.arange(edge_store.edge_index.size(1), dtype=torch.long, device=edge_store.edge_index.device)
        context.state["node_ids"] = node_ids
        context.state["edge_ids"] = edge_ids
        return

    context.state["node_ids_by_type"] = {
        node_type: store.data.get("n_id", torch.arange(
            graph._node_count(node_type),
            dtype=torch.long,
            device=_infer_data_device(store.data),
        ))
        for node_type, store in graph.nodes.items()
    }
    context.state["edge_ids_by_type"] = {
        edge_type: store.data.get(
            "e_id",
            torch.arange(store.edge_index.size(1), dtype=torch.long, device=store.edge_index.device),
        )
        for edge_type, store in graph.edges.items()
    }


def _is_stitched_homo_candidate(graph, source) -> bool:
    return (
        graph is not None
        and graph.schema.time_attr is None
        and set(graph.nodes) == {"node"}
        and len(graph.edges) == 1
        and source is not None
        and graph.nodes["node"].data.get("n_id") is not None
        and all(
            callable(getattr(source, name, None))
            for name in (
                "route_node_ids",
                "partition_node_ids",
                "partition_incident_edge_ids",
                "fetch_partition_incident_edge_index",
                "fetch_node_features",
                "fetch_edge_features",
            )
        )
    )


def _match_partition_shard(graph, source) -> int | None:
    if not _is_stitched_homo_candidate(graph, source):
        return None
    graph_node_ids = torch.as_tensor(graph.nodes["node"].data["n_id"], dtype=torch.long).view(-1)
    try:
        routes = source.route_node_ids(graph_node_ids, node_type="node")
    except Exception:
        return None
    if len(routes) != 1:
        return None
    route = routes[0]
    try:
        partition_node_ids = torch.as_tensor(
            source.partition_node_ids(route.partition_id, node_type="node"),
            dtype=torch.long,
        ).view(-1)
    except Exception:
        return None
    if partition_node_ids.numel() != graph_node_ids.numel():
        return None
    if not torch.equal(torch.sort(partition_node_ids).values, torch.sort(graph_node_ids).values):
        return None
    return int(route.partition_id)


def _match_temporal_partition_shard(graph, source) -> int | None:
    if not (
        graph is not None
        and graph.schema.time_attr is not None
        and set(graph.nodes) == {"node"}
        and len(graph.edges) == 1
        and source is not None
        and graph.nodes["node"].data.get("n_id") is not None
        and all(
            callable(getattr(source, name, None))
            for name in (
                "partition_ids",
                "route_node_ids",
                "partition_node_ids",
                "partition_incident_edge_ids",
                "fetch_partition_incident_edge_index",
                "fetch_node_features",
                "fetch_edge_features",
            )
        )
    ):
        return None
    graph_node_ids = torch.as_tensor(graph.nodes["node"].data["n_id"], dtype=torch.long).view(-1)
    try:
        routes = source.route_node_ids(graph_node_ids, node_type="node")
    except Exception:
        return None
    if len(routes) != 1:
        return None
    route = routes[0]
    try:
        partition_node_ids = torch.as_tensor(
            source.partition_node_ids(route.partition_id, node_type="node"),
            dtype=torch.long,
        ).view(-1)
    except Exception:
        return None
    if partition_node_ids.numel() != graph_node_ids.numel():
        return None
    if not torch.equal(torch.sort(partition_node_ids).values, torch.sort(graph_node_ids).values):
        return None
    return int(route.partition_id)


def _match_hetero_partition_shard(graph, source) -> int | None:
    if not (
        graph is not None
        and graph.schema.time_attr is None
        and not (set(graph.nodes) == {"node"} and len(graph.edges) == 1)
        and source is not None
        and all(store.data.get("n_id") is not None for store in graph.nodes.values())
        and all(
            callable(getattr(source, name, None))
            for name in (
                "route_node_ids",
                "partition_node_ids",
                "partition_incident_edge_ids",
                "fetch_partition_incident_edge_index",
                "fetch_node_features",
                "fetch_edge_features",
            )
        )
    ):
        return None

    partition_id = None
    for node_type, store in graph.nodes.items():
        graph_node_ids = torch.as_tensor(store.data["n_id"], dtype=torch.long).view(-1)
        if graph_node_ids.numel() == 0:
            continue
        try:
            routes = source.route_node_ids(graph_node_ids, node_type=node_type)
        except Exception:
            return None
        if len(routes) != 1:
            return None
        route_partition_id = int(routes[0].partition_id)
        if partition_id is None:
            partition_id = route_partition_id
        elif partition_id != route_partition_id:
            return None

    if partition_id is None:
        return None

    for node_type, store in graph.nodes.items():
        graph_node_ids = torch.as_tensor(store.data["n_id"], dtype=torch.long).view(-1)
        try:
            partition_node_ids = torch.as_tensor(
                source.partition_node_ids(partition_id, node_type=node_type),
                dtype=torch.long,
            ).view(-1)
        except Exception:
            return None
        if partition_node_ids.numel() != graph_node_ids.numel():
            return None
        if not torch.equal(torch.sort(partition_node_ids).values, torch.sort(graph_node_ids).values):
            return None
    return partition_id


def _match_hetero_temporal_partition_shard(graph, source) -> int | None:
    if not (
        graph is not None
        and graph.schema.time_attr is not None
        and not (set(graph.nodes) == {"node"} and len(graph.edges) == 1)
        and source is not None
        and all(store.data.get("n_id") is not None for store in graph.nodes.values())
        and all(
            callable(getattr(source, name, None))
            for name in (
                "partition_ids",
                "route_node_ids",
                "partition_node_ids",
                "partition_incident_edge_ids",
                "fetch_partition_incident_edge_index",
                "fetch_node_features",
                "fetch_edge_features",
            )
        )
    ):
        return None

    partition_id = None
    for node_type, store in graph.nodes.items():
        graph_node_ids = torch.as_tensor(store.data["n_id"], dtype=torch.long).view(-1)
        if graph_node_ids.numel() == 0:
            continue
        try:
            routes = source.route_node_ids(graph_node_ids, node_type=node_type)
        except Exception:
            return None
        if len(routes) != 1:
            return None
        route_partition_id = int(routes[0].partition_id)
        if partition_id is None:
            partition_id = route_partition_id
        elif partition_id != route_partition_id:
            return None

    if partition_id is None:
        return None

    for node_type, store in graph.nodes.items():
        graph_node_ids = torch.as_tensor(store.data["n_id"], dtype=torch.long).view(-1)
        try:
            partition_node_ids = torch.as_tensor(
                source.partition_node_ids(partition_id, node_type=node_type),
                dtype=torch.long,
            ).view(-1)
        except Exception:
            return None
        if partition_node_ids.numel() != graph_node_ids.numel():
            return None
        if not torch.equal(torch.sort(partition_node_ids).values, torch.sort(graph_node_ids).values):
            return None
    return partition_id


def _stitched_hetero_frontier_candidates(
    source,
    frontier_by_type: dict[str, torch.Tensor],
    visited_by_type: dict[str, set[int]],
    *,
    edge_types,
) -> dict[str, list[int]]:
    candidates = {node_type: set() for node_type in visited_by_type}
    for edge_type in edge_types:
        src_type, _, dst_type = edge_type
        src_frontier = torch.as_tensor(frontier_by_type.get(src_type, torch.empty((0,), dtype=torch.long)), dtype=torch.long).view(-1)
        dst_frontier = torch.as_tensor(frontier_by_type.get(dst_type, torch.empty((0,), dtype=torch.long)), dtype=torch.long).view(-1)

        if src_frontier.numel() > 0:
            for route in source.route_node_ids(src_frontier, node_type=src_type):
                edge_index = torch.as_tensor(
                    source.fetch_partition_incident_edge_index(route.partition_id, edge_type=edge_type),
                    dtype=torch.long,
                )
                if edge_index.numel() == 0:
                    continue
                route_ids = route.global_ids.to(dtype=torch.long, device=edge_index.device)
                incident_mask = torch.isin(edge_index[0], route_ids)
                if not bool(incident_mask.any()):
                    continue
                candidates[src_type].update(
                    int(node)
                    for node in edge_index[0, incident_mask].tolist()
                    if int(node) not in visited_by_type[src_type]
                )
                candidates[dst_type].update(
                    int(node)
                    for node in edge_index[1, incident_mask].tolist()
                    if int(node) not in visited_by_type[dst_type]
                )

        if dst_frontier.numel() > 0:
            for route in source.route_node_ids(dst_frontier, node_type=dst_type):
                edge_index = torch.as_tensor(
                    source.fetch_partition_incident_edge_index(route.partition_id, edge_type=edge_type),
                    dtype=torch.long,
                )
                if edge_index.numel() == 0:
                    continue
                route_ids = route.global_ids.to(dtype=torch.long, device=edge_index.device)
                incident_mask = torch.isin(edge_index[1], route_ids)
                if not bool(incident_mask.any()):
                    continue
                candidates[src_type].update(
                    int(node)
                    for node in edge_index[0, incident_mask].tolist()
                    if int(node) not in visited_by_type[src_type]
                )
                candidates[dst_type].update(
                    int(node)
                    for node in edge_index[1, incident_mask].tolist()
                    if int(node) not in visited_by_type[dst_type]
                )

    return {node_type: sorted(values) for node_type, values in candidates.items()}


def _expand_stitched_hetero_global_node_ids(
    source,
    seed_global_ids_by_type: dict[str, torch.Tensor],
    *,
    edge_types,
    fanouts,
    generator=None,
    return_hops: bool = False,
) -> dict[str, torch.Tensor] | tuple[dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    normalized_seeds = {
        node_type: torch.as_tensor(node_ids, dtype=torch.long).view(-1)
        for node_type, node_ids in seed_global_ids_by_type.items()
    }
    device = None
    for seed_ids in normalized_seeds.values():
        if seed_ids.numel() > 0:
            device = seed_ids.device
            break
    if device is None:
        first_seed = next(iter(normalized_seeds.values()), None)
        device = first_seed.device if first_seed is not None else torch.device("cpu")

    visited_by_type = {
        node_type: {int(node) for node in seed_ids.tolist()}
        for node_type, seed_ids in normalized_seeds.items()
    }
    frontier_by_type = {
        node_type: torch.tensor(sorted(node_ids), dtype=torch.long, device=device)
        for node_type, node_ids in visited_by_type.items()
        if node_ids
    }
    hop_nodes = None
    if return_hops:
        hop_nodes = [
            {
                node_type: torch.tensor(sorted(values), dtype=torch.long, device=device)
                for node_type, values in visited_by_type.items()
            }
        ]
    for fanout in fanouts:
        candidate_nodes = _stitched_hetero_frontier_candidates(
            source,
            frontier_by_type,
            visited_by_type,
            edge_types=edge_types,
        )
        next_frontier = {}
        for node_type, values in candidate_nodes.items():
            if fanout != -1 and len(values) > int(fanout):
                permutation = torch.randperm(len(values), generator=generator)[: int(fanout)].tolist()
                values = [values[index] for index in permutation]
            values = sorted(values)
            if not values:
                continue
            frontier_tensor = torch.tensor(values, dtype=torch.long, device=device)
            next_frontier[node_type] = frontier_tensor
            visited_by_type[node_type].update(int(node) for node in frontier_tensor.tolist())
        if hop_nodes is not None:
            hop_nodes.append(
                {
                    node_type: torch.tensor(sorted(values), dtype=torch.long, device=device)
                    for node_type, values in visited_by_type.items()
                }
            )
        elif not next_frontier:
            break
        frontier_by_type = next_frontier

    expanded = {
        node_type: torch.tensor(sorted(values), dtype=torch.long, device=device)
        for node_type, values in visited_by_type.items()
    }
    if hop_nodes is not None:
        return expanded, hop_nodes
    return expanded


def _expand_stitched_hetero_node_ids(
    graph,
    source,
    seed_local_ids: torch.Tensor,
    *,
    seed_node_type: str,
    fanouts,
    generator=None,
    return_hops: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor, dict[str, torch.Tensor], list[dict[str, torch.Tensor]]]:
    seed_local_ids = torch.as_tensor(seed_local_ids, dtype=torch.long).view(-1)
    seed_global_ids = _graph_node_global_ids(graph, seed_local_ids, node_type=seed_node_type)
    seed_global_ids_by_type = {}
    for node_type in graph.schema.node_types:
        if node_type == seed_node_type:
            seed_global_ids_by_type[node_type] = seed_global_ids
            continue
        seed_global_ids_by_type[node_type] = torch.empty(
            (0,),
            dtype=torch.long,
            device=_infer_data_device(graph.nodes[node_type].data),
        )
    node_ids_by_type = _expand_stitched_hetero_global_node_ids(
        source,
        seed_global_ids_by_type,
        edge_types=tuple(graph.edges),
        fanouts=fanouts,
        generator=generator,
        return_hops=return_hops,
    )
    if return_hops:
        node_ids_by_type, node_hops_by_type = node_ids_by_type
        return seed_global_ids, node_ids_by_type, node_hops_by_type
    return seed_global_ids, node_ids_by_type


def _stitched_hetero_link_seed_global_ids(graph, records) -> dict[str, torch.Tensor]:
    seed_local_ids_by_type = {node_type: set() for node_type in graph.schema.node_types}
    for record in records:
        src_type, _, dst_type = _resolve_link_record_edge_type(record)
        seed_local_ids_by_type[src_type].add(int(record.src_index))
        seed_local_ids_by_type[dst_type].add(int(record.dst_index))

    seed_global_ids_by_type = {}
    for node_type in graph.schema.node_types:
        local_ids = sorted(seed_local_ids_by_type[node_type])
        device = _infer_data_device(graph.nodes[node_type].data)
        local_tensor = torch.tensor(local_ids, dtype=torch.long, device=device)
        seed_global_ids_by_type[node_type] = _graph_node_global_ids(
            graph,
            local_tensor,
            node_type=node_type,
        )
    return seed_global_ids_by_type


def _collect_stitched_hetero_edges(
    source,
    node_ids_by_type: dict[str, torch.Tensor],
    *,
    edge_types,
) -> tuple[dict[tuple[str, str, str], torch.Tensor], dict[tuple[str, str, str], torch.Tensor]]:
    edge_ids_by_type = {}
    edge_index_by_type = {}
    for edge_type in edge_types:
        src_type, _, dst_type = edge_type
        src_nodes = torch.as_tensor(node_ids_by_type[src_type], dtype=torch.long).view(-1)
        dst_nodes = torch.as_tensor(node_ids_by_type[dst_type], dtype=torch.long).view(-1)
        device = src_nodes.device if src_nodes.numel() > 0 else dst_nodes.device
        visited_src = {int(node) for node in src_nodes.tolist()}
        visited_dst = {int(node) for node in dst_nodes.tolist()}
        edge_records: dict[int, tuple[int, int]] = {}

        if src_nodes.numel() > 0:
            for route in source.route_node_ids(src_nodes, node_type=src_type):
                edge_ids = torch.as_tensor(
                    source.partition_incident_edge_ids(route.partition_id, edge_type=edge_type),
                    dtype=torch.long,
                ).view(-1)
                edge_index = torch.as_tensor(
                    source.fetch_partition_incident_edge_index(route.partition_id, edge_type=edge_type),
                    dtype=torch.long,
                )
                if edge_ids.numel() == 0 or edge_index.numel() == 0:
                    continue
                route_ids = route.global_ids.to(dtype=torch.long, device=edge_index.device)
                incident_mask = torch.isin(edge_index[0], route_ids)
                if not bool(incident_mask.any()):
                    continue
                for edge_id, src, dst in zip(
                    edge_ids[incident_mask].tolist(),
                    edge_index[0, incident_mask].tolist(),
                    edge_index[1, incident_mask].tolist(),
                ):
                    src = int(src)
                    dst = int(dst)
                    if src in visited_src and dst in visited_dst:
                        edge_records[int(edge_id)] = (src, dst)

        if dst_nodes.numel() > 0:
            for route in source.route_node_ids(dst_nodes, node_type=dst_type):
                edge_ids = torch.as_tensor(
                    source.partition_incident_edge_ids(route.partition_id, edge_type=edge_type),
                    dtype=torch.long,
                ).view(-1)
                edge_index = torch.as_tensor(
                    source.fetch_partition_incident_edge_index(route.partition_id, edge_type=edge_type),
                    dtype=torch.long,
                )
                if edge_ids.numel() == 0 or edge_index.numel() == 0:
                    continue
                route_ids = route.global_ids.to(dtype=torch.long, device=edge_index.device)
                incident_mask = torch.isin(edge_index[1], route_ids)
                if not bool(incident_mask.any()):
                    continue
                for edge_id, src, dst in zip(
                    edge_ids[incident_mask].tolist(),
                    edge_index[0, incident_mask].tolist(),
                    edge_index[1, incident_mask].tolist(),
                ):
                    src = int(src)
                    dst = int(dst)
                    if src in visited_src and dst in visited_dst:
                        edge_records[int(edge_id)] = (src, dst)

        if not edge_records:
            edge_ids_by_type[edge_type] = torch.empty((0,), dtype=torch.long, device=device)
            edge_index_by_type[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)
            continue

        ordered_edge_ids = sorted(edge_records)
        edge_ids_by_type[edge_type] = torch.tensor(ordered_edge_ids, dtype=torch.long, device=device)
        edge_index_by_type[edge_type] = torch.tensor(
            [
                [edge_records[edge_id][0] for edge_id in ordered_edge_ids],
                [edge_records[edge_id][1] for edge_id in ordered_edge_ids],
            ],
            dtype=torch.long,
            device=device,
        )

    return edge_ids_by_type, edge_index_by_type


def _relabel_stitched_edge_index_by_type(
    node_ids_by_type: dict[str, torch.Tensor],
    edge_index_by_type: dict[tuple[str, str, str], torch.Tensor],
) -> dict[tuple[str, str, str], torch.Tensor]:
    relabeled = {}
    for edge_type, edge_index_global in edge_index_by_type.items():
        src_type, _, dst_type = edge_type
        src_positions = {int(node_id): index for index, node_id in enumerate(torch.as_tensor(node_ids_by_type[src_type], dtype=torch.long).tolist())}
        dst_positions = {int(node_id): index for index, node_id in enumerate(torch.as_tensor(node_ids_by_type[dst_type], dtype=torch.long).tolist())}
        edge_index_global = torch.as_tensor(edge_index_global, dtype=torch.long)
        device = torch.as_tensor(node_ids_by_type[src_type], dtype=torch.long).device
        if edge_index_global.numel() == 0:
            relabeled[edge_type] = torch.empty((2, 0), dtype=torch.long, device=device)
            continue
        relabeled[edge_type] = torch.tensor(
            [
                [src_positions[int(node_id)] for node_id in edge_index_global[0].tolist()],
                [dst_positions[int(node_id)] for node_id in edge_index_global[1].tolist()],
            ],
            dtype=torch.long,
            device=device,
        )
    return relabeled


def _stitched_frontier_candidates(source, frontier_global_ids: torch.Tensor, visited: set[int], *, edge_type) -> list[int]:
    frontier_global_ids = torch.as_tensor(frontier_global_ids, dtype=torch.long).view(-1)
    if frontier_global_ids.numel() == 0:
        return []
    candidates = set()
    for route in source.route_node_ids(frontier_global_ids, node_type="node"):
        edge_index = torch.as_tensor(
            source.fetch_partition_incident_edge_index(route.partition_id, edge_type=edge_type),
            dtype=torch.long,
        )
        if edge_index.numel() == 0:
            continue
        route_ids = route.global_ids.to(dtype=torch.long, device=edge_index.device)
        incident_mask = torch.isin(edge_index[0], route_ids) | torch.isin(edge_index[1], route_ids)
        if not incident_mask.any():
            continue
        for node in torch.unique(edge_index[:, incident_mask]).tolist():
            node = int(node)
            if node not in visited:
                candidates.add(node)
    return sorted(candidates)


def _expand_stitched_homo_global_node_ids(
    source,
    seed_global_ids: torch.Tensor,
    *,
    fanouts,
    edge_type,
    generator=None,
    return_hops: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    seed_global_ids = torch.as_tensor(seed_global_ids, dtype=torch.long).view(-1)
    visited = {int(node) for node in seed_global_ids.tolist()}
    frontier = torch.tensor(sorted(visited), dtype=torch.long, device=seed_global_ids.device)
    hop_nodes = None
    if return_hops:
        hop_nodes = [torch.tensor(sorted(visited), dtype=torch.long, device=seed_global_ids.device)]
    for fanout in fanouts:
        candidate_nodes = _stitched_frontier_candidates(source, frontier, visited, edge_type=edge_type)
        if fanout != -1 and len(candidate_nodes) > int(fanout):
            permutation = torch.randperm(len(candidate_nodes), generator=generator)[: int(fanout)].tolist()
            candidate_nodes = [candidate_nodes[index] for index in permutation]
        frontier = torch.tensor(candidate_nodes, dtype=torch.long, device=seed_global_ids.device)
        visited.update(int(node) for node in frontier.tolist())
        if hop_nodes is not None:
            hop_nodes.append(torch.tensor(sorted(visited), dtype=torch.long, device=seed_global_ids.device))
        elif frontier.numel() == 0:
            break
    expanded = torch.tensor(sorted(visited), dtype=torch.long, device=seed_global_ids.device)
    if hop_nodes is not None:
        return expanded, hop_nodes
    return expanded


def _expand_stitched_homo_node_ids(
    graph,
    source,
    seed_local_ids: torch.Tensor,
    *,
    fanouts,
    return_hops: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    seed_local_ids = torch.as_tensor(seed_local_ids, dtype=torch.long).view(-1)
    seed_global_ids = _graph_node_global_ids(graph, seed_local_ids, node_type="node")
    expanded = _expand_stitched_homo_global_node_ids(
        source,
        seed_global_ids,
        fanouts=fanouts,
        edge_type=graph._default_edge_type(),
        return_hops=return_hops,
    )
    if return_hops:
        node_ids_global, node_hops = expanded
        return seed_global_ids, node_ids_global, node_hops
    return seed_global_ids, expanded


def _collect_stitched_homo_edges(source, node_ids_global: torch.Tensor, *, edge_type) -> tuple[torch.Tensor, torch.Tensor]:
    node_ids_global = torch.as_tensor(node_ids_global, dtype=torch.long).view(-1)
    visited = {int(node) for node in node_ids_global.tolist()}
    edge_records: dict[int, tuple[int, int]] = {}
    for route in source.route_node_ids(node_ids_global, node_type="node"):
        edge_ids = torch.as_tensor(
            source.partition_incident_edge_ids(route.partition_id, edge_type=edge_type),
            dtype=torch.long,
        ).view(-1)
        edge_index = torch.as_tensor(
            source.fetch_partition_incident_edge_index(route.partition_id, edge_type=edge_type),
            dtype=torch.long,
        )
        if edge_ids.numel() == 0 or edge_index.numel() == 0:
            continue
        route_ids = route.global_ids.to(dtype=torch.long, device=edge_index.device)
        incident_mask = torch.isin(edge_index[0], route_ids) | torch.isin(edge_index[1], route_ids)
        if not incident_mask.any():
            continue
        current_edge_ids = edge_ids[incident_mask]
        current_edge_index = edge_index[:, incident_mask]
        for edge_id, src, dst in zip(
            current_edge_ids.tolist(),
            current_edge_index[0].tolist(),
            current_edge_index[1].tolist(),
        ):
            src = int(src)
            dst = int(dst)
            if src in visited and dst in visited:
                edge_records[int(edge_id)] = (src, dst)
    if not edge_records:
        return (
            torch.empty((0,), dtype=torch.long, device=node_ids_global.device),
            torch.empty((2, 0), dtype=torch.long, device=node_ids_global.device),
        )
    ordered_edge_ids = sorted(edge_records)
    edge_ids_global = torch.tensor(ordered_edge_ids, dtype=torch.long, device=node_ids_global.device)
    edge_index_global = torch.tensor(
        [
            [edge_records[edge_id][0] for edge_id in ordered_edge_ids],
            [edge_records[edge_id][1] for edge_id in ordered_edge_ids],
        ],
        dtype=torch.long,
        device=node_ids_global.device,
    )
    return edge_ids_global, edge_index_global


def _relabel_stitched_edge_index(node_ids_global: torch.Tensor, edge_index_global: torch.Tensor) -> torch.Tensor:
    node_ids_global = torch.as_tensor(node_ids_global, dtype=torch.long).view(-1)
    edge_index_global = torch.as_tensor(edge_index_global, dtype=torch.long)
    if edge_index_global.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=node_ids_global.device)
    positions = {int(node_id): index for index, node_id in enumerate(node_ids_global.tolist())}
    return torch.tensor(
        [
            [positions[int(node_id)] for node_id in edge_index_global[0].tolist()],
            [positions[int(node_id)] for node_id in edge_index_global[1].tolist()],
        ],
        dtype=torch.long,
        device=node_ids_global.device,
    )


def _fetch_stitched_homo_node_data(graph, source, node_ids_global: torch.Tensor) -> dict[str, Any]:
    node_ids_global = torch.as_tensor(node_ids_global, dtype=torch.long).view(-1)
    node_store = graph.nodes["node"]
    node_count = graph._node_count("node")
    node_data = {"n_id": node_ids_global}
    for key, value in node_store.data.items():
        if key == "n_id":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == node_count:
            node_data[key] = source.fetch_node_features(("node", "node", key), node_ids_global).values
        else:
            node_data[key] = value
    return node_data


def _fetch_routed_edge_feature_values(source, key, edge_ids: torch.Tensor) -> torch.Tensor:
    try:
        return source.fetch_edge_features(key, edge_ids).values
    except KeyError:
        return _fallback_routed_edge_tensor_slice(source, key, edge_ids).values


def _fallback_routed_edge_tensor_slice(source, key, edge_ids: torch.Tensor) -> TensorSlice:
    _, edge_type, feature_name = key
    edge_type = tuple(edge_type)
    edge_ids = torch.as_tensor(edge_ids, dtype=torch.long).view(-1)
    shards = getattr(source, "shards", None)
    if not isinstance(shards, dict):
        raise KeyError(f"routed edge feature fallback is unavailable for key {key!r}")

    template = None
    for shard in shards.values():
        local_feature = shard.graph.edges[edge_type].data.get(feature_name)
        if isinstance(local_feature, torch.Tensor):
            template = local_feature
            break
        boundary_feature = shard.boundary_edge_data_by_type.get(edge_type, {}).get(feature_name)
        if isinstance(boundary_feature, torch.Tensor):
            template = boundary_feature
            break
    if template is None:
        raise KeyError(f"missing routed edge feature {feature_name!r} for edge type {edge_type!r}")
    if edge_ids.numel() == 0:
        return TensorSlice(index=edge_ids, values=template.new_empty((0,) + tuple(template.shape[1:])))

    values = template.new_empty((edge_ids.numel(),) + tuple(template.shape[1:]))
    remaining = {int(edge_id): index for index, edge_id in enumerate(edge_ids.tolist())}
    for shard in shards.values():
        local_feature = shard.graph.edges[edge_type].data.get(feature_name)
        local_edge_ids = shard.edge_ids(edge_type=edge_type)
        if isinstance(local_feature, torch.Tensor) and local_feature.ndim > 0 and local_feature.size(0) == local_edge_ids.numel():
            for current_index, edge_id in enumerate(local_edge_ids.tolist()):
                target_index = remaining.pop(int(edge_id), None)
                if target_index is None:
                    continue
                values[target_index] = local_feature[current_index]

        boundary_data = shard.boundary_edge_data_by_type.get(edge_type, {})
        boundary_feature = boundary_data.get(feature_name)
        boundary_edge_ids = torch.as_tensor(boundary_data.get("e_id", torch.empty((0,), dtype=torch.long)), dtype=torch.long).view(-1)
        if isinstance(boundary_feature, torch.Tensor) and boundary_feature.ndim > 0 and boundary_feature.size(0) == boundary_edge_ids.numel():
            for current_index, edge_id in enumerate(boundary_edge_ids.tolist()):
                target_index = remaining.pop(int(edge_id), None)
                if target_index is None:
                    continue
                values[target_index] = boundary_feature[current_index]
        if not remaining:
            break

    if remaining:
        missing = ", ".join(str(edge_id) for edge_id in sorted(remaining))
        raise KeyError(f"missing routed edge feature {feature_name!r} for edge ids: {missing}")
    return TensorSlice(index=edge_ids, values=values)


def _fetch_stitched_homo_edge_feature(graph, source, edge_ids_global: torch.Tensor, *, edge_type, feature_name: str) -> torch.Tensor:
    edge_ids_global = torch.as_tensor(edge_ids_global, dtype=torch.long).view(-1)
    edge_store = graph.edges[tuple(edge_type)]
    template = edge_store.data[feature_name]
    if edge_ids_global.numel() == 0:
        return template.new_empty((0,) + tuple(template.shape[1:]))
    try:
        return source.fetch_edge_features(("edge", tuple(edge_type), feature_name), edge_ids_global).values
    except KeyError:
        return _fallback_routed_edge_tensor_slice(
            source,
            ("edge", tuple(edge_type), feature_name),
            edge_ids_global,
        ).values


def _fetch_stitched_homo_edge_data(graph, source, edge_ids_global: torch.Tensor, *, edge_type) -> dict[str, Any]:
    edge_ids_global = torch.as_tensor(edge_ids_global, dtype=torch.long).view(-1)
    edge_store = graph.edges[tuple(edge_type)]
    edge_count = int(edge_store.edge_index.size(1))
    edge_data = {"e_id": edge_ids_global}
    for key, value in edge_store.data.items():
        if key in {"edge_index", "e_id"}:
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = _fetch_stitched_homo_edge_feature(
                graph,
                source,
                edge_ids_global,
                edge_type=edge_type,
                feature_name=key,
            )
        else:
            edge_data[key] = value
    return edge_data


def _build_stitched_homo_graph(graph, source, node_ids_global: torch.Tensor, edge_ids_global: torch.Tensor, edge_index: torch.Tensor) -> Graph:
    stitched_graph = Graph.homo(
        edge_index=edge_index,
        edge_data=_fetch_stitched_homo_edge_data(graph, source, edge_ids_global, edge_type=graph._default_edge_type()),
        **_fetch_stitched_homo_node_data(graph, source, node_ids_global),
    )
    stitched_graph.feature_store = source
    return stitched_graph


def _build_stitched_node_samples(
    context: "MaterializationContext",
    stitched_graph: Graph,
    seed_local_ids: torch.Tensor,
    seed_global_ids: torch.Tensor,
):
    metadata = dict(getattr(context.request, "metadata", {}))
    sample_id = context.metadata.get("sample_id", metadata.get("sample_id"))
    source_graph_id = context.metadata.get("source_graph_id", metadata.get("source_graph_id"))
    seed_positions = {int(node_id): index for index, node_id in enumerate(stitched_graph.n_id.tolist())}
    samples = []
    for seed_local, seed_global in zip(
        torch.as_tensor(seed_local_ids, dtype=torch.long).view(-1).tolist(),
        torch.as_tensor(seed_global_ids, dtype=torch.long).view(-1).tolist(),
    ):
        sample_metadata = dict(metadata)
        sample_metadata["seed"] = int(seed_local)
        samples.append(
            SampleRecord(
                graph=stitched_graph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                subgraph_seed=seed_positions[int(seed_global)],
            )
        )
    if len(samples) == 1:
        return samples[0]
    return samples


def _fetch_stitched_node_data(graph, source, node_ids_global: torch.Tensor, *, node_type: str) -> dict[str, Any]:
    node_ids_global = torch.as_tensor(node_ids_global, dtype=torch.long).view(-1)
    node_store = graph.nodes[str(node_type)]
    node_count = graph._node_count(str(node_type))
    node_data = {"n_id": node_ids_global}
    for key, value in node_store.data.items():
        if key == "n_id":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == node_count:
            node_data[key] = source.fetch_node_features(("node", str(node_type), key), node_ids_global).values
        else:
            node_data[key] = value
    return node_data


def _build_stitched_hetero_graph(
    graph,
    source,
    node_ids_by_type: dict[str, torch.Tensor],
    edge_ids_by_type: dict[tuple[str, str, str], torch.Tensor],
    edge_index_by_type: dict[tuple[str, str, str], torch.Tensor],
) -> Graph:
    nodes = {
        node_type: _fetch_stitched_node_data(graph, source, node_ids_by_type[node_type], node_type=node_type)
        for node_type in graph.nodes
    }
    edges = {
        edge_type: {
            "edge_index": edge_index_by_type[edge_type],
            **_fetch_stitched_homo_edge_data(graph, source, edge_ids_by_type[edge_type], edge_type=edge_type),
        }
        for edge_type in graph.edges
    }
    stitched_graph = Graph.hetero(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr)
    stitched_graph.feature_store = source
    return stitched_graph


def _build_stitched_hetero_node_samples(
    context: "MaterializationContext",
    stitched_graph: Graph,
    seed_local_ids: torch.Tensor,
    seed_global_ids: torch.Tensor,
    *,
    node_type: str,
):
    metadata = dict(getattr(context.request, "metadata", {}))
    sample_id = context.metadata.get("sample_id", metadata.get("sample_id"))
    source_graph_id = context.metadata.get("source_graph_id", metadata.get("source_graph_id"))
    seed_positions = {
        int(node_id): index
        for index, node_id in enumerate(stitched_graph.nodes[str(node_type)].data["n_id"].tolist())
    }
    samples = []
    for seed_local, seed_global in zip(
        torch.as_tensor(seed_local_ids, dtype=torch.long).view(-1).tolist(),
        torch.as_tensor(seed_global_ids, dtype=torch.long).view(-1).tolist(),
    ):
        sample_metadata = dict(metadata)
        sample_metadata["seed"] = int(seed_local)
        sample_metadata.setdefault("node_type", str(node_type))
        samples.append(
            SampleRecord(
                graph=stitched_graph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                subgraph_seed=seed_positions[int(seed_global)],
            )
        )
    if len(samples) == 1:
        return samples[0]
    return samples


def _build_stitched_homo_temporal_graph(
    graph,
    source,
    node_ids_global: torch.Tensor,
    edge_ids_global: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_type,
) -> Graph:
    stitched_graph = Graph.temporal(
        nodes={"node": _fetch_stitched_homo_node_data(graph, source, node_ids_global)},
        edges={
            tuple(edge_type): {
                "edge_index": edge_index,
                **_fetch_stitched_homo_edge_data(graph, source, edge_ids_global, edge_type=edge_type),
            }
        },
        time_attr=graph.schema.time_attr,
    )
    stitched_graph.feature_store = source
    return stitched_graph


def _build_stitched_hetero_temporal_graph(
    graph,
    source,
    node_ids_by_type: dict[str, torch.Tensor],
    edge_ids_global: torch.Tensor,
    edge_index: torch.Tensor,
    *,
    edge_type,
) -> Graph:
    src_type, _, dst_type = tuple(edge_type)
    unique_node_types = tuple(dict.fromkeys((src_type, dst_type)))
    stitched_graph = Graph.temporal(
        nodes={
            node_type: _fetch_stitched_node_data(graph, source, node_ids_by_type[node_type], node_type=node_type)
            for node_type in unique_node_types
        },
        edges={
            tuple(edge_type): {
                "edge_index": edge_index,
                **_fetch_stitched_homo_edge_data(graph, source, edge_ids_global, edge_type=edge_type),
            }
        },
        time_attr=graph.schema.time_attr,
    )
    stitched_graph.feature_store = source
    return stitched_graph


def _build_stitched_homo_temporal_record(graph, record, stitched_graph: Graph) -> TemporalEventRecord:
    seed_positions = {int(node_id): index for index, node_id in enumerate(stitched_graph.n_id.tolist())}
    src_global = int(_graph_node_global_ids(graph, torch.tensor([record.src_index]), node_type="node").item())
    dst_global = int(_graph_node_global_ids(graph, torch.tensor([record.dst_index]), node_type="node").item())
    edge_type = _resolve_temporal_record_edge_type(record)
    return TemporalEventRecord(
        graph=stitched_graph,
        src_index=seed_positions[src_global],
        dst_index=seed_positions[dst_global],
        timestamp=int(record.timestamp),
        label=int(record.label),
        event_features=record.event_features,
        metadata=dict(record.metadata),
        sample_id=record.sample_id,
        edge_type=edge_type,
    )


def _build_stitched_hetero_temporal_record(graph, record, stitched_graph: Graph) -> TemporalEventRecord:
    edge_type = _resolve_temporal_record_edge_type(record)
    src_type, _, dst_type = edge_type
    seed_positions_by_type = {
        node_type: {
            int(node_id): index
            for index, node_id in enumerate(stitched_graph.nodes[node_type].data["n_id"].tolist())
        }
        for node_type in stitched_graph.nodes
    }
    src_global = int(
        _graph_node_global_ids(
            graph,
            torch.tensor([record.src_index], dtype=torch.long),
            node_type=src_type,
        ).item()
    )
    dst_global = int(
        _graph_node_global_ids(
            graph,
            torch.tensor([record.dst_index], dtype=torch.long),
            node_type=dst_type,
        ).item()
    )
    return TemporalEventRecord(
        graph=stitched_graph,
        src_index=seed_positions_by_type[src_type][src_global],
        dst_index=seed_positions_by_type[dst_type][dst_global],
        timestamp=int(record.timestamp),
        label=int(record.label),
        event_features=record.event_features,
        metadata=dict(record.metadata),
        sample_id=record.sample_id,
        edge_type=edge_type,
    )


def _build_stitched_homo_temporal_history(
    graph,
    source,
    sampler,
    record,
    *,
    edge_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    partition_ids = getattr(source, "partition_ids", None)
    if not callable(partition_ids):
        raise ValueError("stitched temporal sampling requires coordinator partition access")
    time_attr = graph.schema.time_attr
    if time_attr is None:
        raise ValueError("stitched temporal sampling requires a temporal graph")

    timestamp = int(record.timestamp)
    edge_records: dict[int, tuple[int, int, int]] = {}
    for partition_id in partition_ids():
        shard_edge_ids = torch.as_tensor(
            source.partition_incident_edge_ids(partition_id, edge_type=edge_type),
            dtype=torch.long,
        ).view(-1)
        if shard_edge_ids.numel() == 0:
            continue

        shard_edge_index = torch.as_tensor(
            source.fetch_partition_incident_edge_index(partition_id, edge_type=edge_type),
            dtype=torch.long,
        )
        shard_timestamps = torch.as_tensor(
            _fetch_routed_edge_feature_values(
                source,
                ("edge", tuple(edge_type), time_attr),
                shard_edge_ids,
            )
        ).view(-1)

        if sampler.strict_history:
            edge_mask = shard_timestamps < timestamp
        else:
            edge_mask = shard_timestamps <= timestamp
        if sampler.time_window is not None:
            edge_mask &= shard_timestamps >= (timestamp - sampler.time_window)
        if not bool(edge_mask.any()):
            continue

        for edge_id, src, dst, current_timestamp in zip(
            shard_edge_ids[edge_mask].tolist(),
            shard_edge_index[0, edge_mask].tolist(),
            shard_edge_index[1, edge_mask].tolist(),
            shard_timestamps[edge_mask].tolist(),
        ):
            edge_records[int(edge_id)] = (int(src), int(dst), int(current_timestamp))

    device = graph.edges[tuple(edge_type)].edge_index.device
    if not edge_records:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((2, 0), dtype=torch.long, device=device),
        )

    ordered_edge_ids = sorted(edge_records)
    edge_ids_global = torch.tensor(ordered_edge_ids, dtype=torch.long, device=device)
    edge_index_global = torch.tensor(
        [
            [edge_records[edge_id][0] for edge_id in ordered_edge_ids],
            [edge_records[edge_id][1] for edge_id in ordered_edge_ids],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_timestamps = torch.tensor(
        [edge_records[edge_id][2] for edge_id in ordered_edge_ids],
        dtype=torch.long,
        device=device,
    )

    if sampler.max_events is not None and edge_ids_global.numel() > sampler.max_events:
        time_order = torch.argsort(edge_timestamps, stable=True)
        edge_ids_global = edge_ids_global[time_order][-sampler.max_events :]
        edge_index_global = edge_index_global[:, time_order][:, -sampler.max_events :]

    return edge_ids_global, edge_index_global


def _build_stitched_hetero_temporal_history(
    graph,
    source,
    sampler,
    record,
    *,
    edge_type,
) -> tuple[torch.Tensor, torch.Tensor]:
    partition_ids = getattr(source, "partition_ids", None)
    if not callable(partition_ids):
        raise ValueError("stitched temporal sampling requires coordinator partition access")
    time_attr = graph.schema.time_attr
    if time_attr is None:
        raise ValueError("stitched temporal sampling requires a temporal graph")

    timestamp = int(record.timestamp)
    edge_records: dict[int, tuple[int, int, int]] = {}
    for partition_id in partition_ids():
        shard_edge_ids = torch.as_tensor(
            source.partition_incident_edge_ids(partition_id, edge_type=edge_type),
            dtype=torch.long,
        ).view(-1)
        if shard_edge_ids.numel() == 0:
            continue

        shard_edge_index = torch.as_tensor(
            source.fetch_partition_incident_edge_index(partition_id, edge_type=edge_type),
            dtype=torch.long,
        )
        shard_timestamps = torch.as_tensor(
            _fetch_routed_edge_feature_values(
                source,
                ("edge", tuple(edge_type), time_attr),
                shard_edge_ids,
            )
        ).view(-1)

        if sampler.strict_history:
            edge_mask = shard_timestamps < timestamp
        else:
            edge_mask = shard_timestamps <= timestamp
        if sampler.time_window is not None:
            edge_mask &= shard_timestamps >= (timestamp - sampler.time_window)
        if not bool(edge_mask.any()):
            continue

        for edge_id, src, dst, current_timestamp in zip(
            shard_edge_ids[edge_mask].tolist(),
            shard_edge_index[0, edge_mask].tolist(),
            shard_edge_index[1, edge_mask].tolist(),
            shard_timestamps[edge_mask].tolist(),
        ):
            edge_records[int(edge_id)] = (int(src), int(dst), int(current_timestamp))

    device = graph.edges[tuple(edge_type)].edge_index.device
    if not edge_records:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((2, 0), dtype=torch.long, device=device),
        )

    ordered_edge_ids = sorted(edge_records)
    edge_ids_global = torch.tensor(ordered_edge_ids, dtype=torch.long, device=device)
    edge_index_global = torch.tensor(
        [
            [edge_records[edge_id][0] for edge_id in ordered_edge_ids],
            [edge_records[edge_id][1] for edge_id in ordered_edge_ids],
        ],
        dtype=torch.long,
        device=device,
    )
    edge_timestamps = torch.tensor(
        [edge_records[edge_id][2] for edge_id in ordered_edge_ids],
        dtype=torch.long,
        device=device,
    )

    if sampler.max_events is not None and edge_ids_global.numel() > sampler.max_events:
        time_order = torch.argsort(edge_timestamps, stable=True)
        edge_ids_global = edge_ids_global[time_order][-sampler.max_events :]
        edge_index_global = edge_index_global[:, time_order][:, -sampler.max_events :]

    return edge_ids_global, edge_index_global


def _stitched_hetero_temporal_seed_global_ids(graph, record, *, edge_type) -> dict[str, torch.Tensor]:
    src_type, _, dst_type = tuple(edge_type)
    unique_node_types = tuple(dict.fromkeys((src_type, dst_type)))
    seed_local_ids_by_type = {node_type: set() for node_type in unique_node_types}
    seed_local_ids_by_type[src_type].add(int(record.src_index))
    seed_local_ids_by_type[dst_type].add(int(record.dst_index))

    seed_global_ids_by_type = {}
    for node_type in unique_node_types:
        local_ids = sorted(seed_local_ids_by_type[node_type])
        local_tensor = torch.tensor(
            local_ids,
            dtype=torch.long,
            device=_infer_data_device(graph.nodes[node_type].data),
        )
        seed_global_ids_by_type[node_type] = _graph_node_global_ids(
            graph,
            local_tensor,
            node_type=node_type,
        )
    return seed_global_ids_by_type


def _expand_stitched_homo_temporal_node_ids(
    seed_global_ids: torch.Tensor,
    history_edge_index_global: torch.Tensor,
    *,
    fanouts,
    generator=None,
) -> torch.Tensor:
    seed_global_ids = torch.as_tensor(seed_global_ids, dtype=torch.long).view(-1)
    history_edge_index_global = torch.as_tensor(history_edge_index_global, dtype=torch.long)
    visited = {int(node) for node in seed_global_ids.tolist()}
    frontier = torch.tensor(sorted(visited), dtype=torch.long, device=seed_global_ids.device)
    for fanout in fanouts:
        if frontier.numel() == 0 or history_edge_index_global.numel() == 0:
            break
        frontier_tensor = frontier.to(device=history_edge_index_global.device)
        incident_mask = torch.isin(history_edge_index_global[0], frontier_tensor) | torch.isin(
            history_edge_index_global[1], frontier_tensor
        )
        if not bool(incident_mask.any()):
            break
        candidate_nodes = [
            int(node)
            for node in torch.unique(history_edge_index_global[:, incident_mask]).tolist()
            if int(node) not in visited
        ]
        if fanout != -1 and len(candidate_nodes) > int(fanout):
            permutation = torch.randperm(len(candidate_nodes), generator=generator)[: int(fanout)].tolist()
            candidate_nodes = [candidate_nodes[index] for index in permutation]
        frontier = torch.tensor(sorted(candidate_nodes), dtype=torch.long, device=seed_global_ids.device)
        visited.update(int(node) for node in frontier.tolist())
    return torch.tensor(sorted(visited), dtype=torch.long, device=seed_global_ids.device)


def _expand_stitched_hetero_temporal_node_ids(
    seed_global_ids_by_type: dict[str, torch.Tensor],
    history_edge_index_global: torch.Tensor,
    *,
    src_type: str,
    dst_type: str,
    fanouts,
    generator=None,
) -> dict[str, torch.Tensor]:
    history_edge_index_global = torch.as_tensor(history_edge_index_global, dtype=torch.long)
    unique_node_types = tuple(dict.fromkeys((src_type, dst_type)))
    device = None
    for node_type in unique_node_types:
        seed_ids = torch.as_tensor(seed_global_ids_by_type[node_type], dtype=torch.long).view(-1)
        if seed_ids.numel() > 0:
            device = seed_ids.device
            break
    if device is None:
        device = history_edge_index_global.device if history_edge_index_global.numel() > 0 else torch.device("cpu")

    visited = {
        node_type: {int(node) for node in torch.as_tensor(seed_global_ids_by_type[node_type], dtype=torch.long).view(-1).tolist()}
        for node_type in unique_node_types
    }
    frontier = {node_type: set(node_ids) for node_type, node_ids in visited.items()}
    for fanout in fanouts:
        if history_edge_index_global.numel() == 0:
            break
        incident_mask = torch.zeros(history_edge_index_global.size(1), dtype=torch.bool, device=history_edge_index_global.device)
        src_frontier = frontier.get(src_type, set())
        dst_frontier = frontier.get(dst_type, set())
        if src_frontier:
            src_tensor = torch.tensor(sorted(src_frontier), dtype=torch.long, device=history_edge_index_global.device)
            incident_mask |= torch.isin(history_edge_index_global[0], src_tensor)
        if dst_frontier:
            dst_tensor = torch.tensor(sorted(dst_frontier), dtype=torch.long, device=history_edge_index_global.device)
            incident_mask |= torch.isin(history_edge_index_global[1], dst_tensor)
        if not bool(incident_mask.any()):
            break

        candidates = {node_type: set() for node_type in unique_node_types}
        candidates[src_type].update(
            int(node)
            for node in history_edge_index_global[0, incident_mask].tolist()
            if int(node) not in visited[src_type]
        )
        candidates[dst_type].update(
            int(node)
            for node in history_edge_index_global[1, incident_mask].tolist()
            if int(node) not in visited[dst_type]
        )

        next_frontier = {}
        for node_type, values in candidates.items():
            candidate_list = sorted(values)
            if fanout != -1 and len(candidate_list) > int(fanout):
                permutation = torch.randperm(len(candidate_list), generator=generator)[: int(fanout)].tolist()
                candidate_list = [candidate_list[index] for index in permutation]
            next_frontier[node_type] = set(candidate_list)

        for node_type, node_ids in next_frontier.items():
            visited[node_type].update(node_ids)
        if not any(next_frontier.values()):
            break
        frontier = next_frontier

    return {
        node_type: torch.tensor(sorted(node_ids), dtype=torch.long, device=device)
        for node_type, node_ids in visited.items()
    }


def _induce_stitched_homo_temporal_edges(
    node_ids_global: torch.Tensor,
    edge_ids_global: torch.Tensor,
    edge_index_global: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    node_ids_global = torch.as_tensor(node_ids_global, dtype=torch.long).view(-1)
    edge_ids_global = torch.as_tensor(edge_ids_global, dtype=torch.long).view(-1)
    edge_index_global = torch.as_tensor(edge_index_global, dtype=torch.long)
    if edge_ids_global.numel() == 0 or edge_index_global.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=node_ids_global.device),
            torch.empty((2, 0), dtype=torch.long, device=node_ids_global.device),
        )
    edge_mask = torch.isin(edge_index_global[0], node_ids_global) & torch.isin(edge_index_global[1], node_ids_global)
    return edge_ids_global[edge_mask], edge_index_global[:, edge_mask]


def _induce_stitched_hetero_temporal_edges(
    node_ids_by_type: dict[str, torch.Tensor],
    edge_ids_global: torch.Tensor,
    edge_index_global: torch.Tensor,
    *,
    src_type: str,
    dst_type: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    edge_ids_global = torch.as_tensor(edge_ids_global, dtype=torch.long).view(-1)
    edge_index_global = torch.as_tensor(edge_index_global, dtype=torch.long)
    src_node_ids = torch.as_tensor(node_ids_by_type[src_type], dtype=torch.long).view(-1)
    dst_node_ids = torch.as_tensor(node_ids_by_type[dst_type], dtype=torch.long).view(-1)
    device = src_node_ids.device if src_node_ids.numel() > 0 else dst_node_ids.device
    if edge_ids_global.numel() == 0 or edge_index_global.numel() == 0:
        return (
            torch.empty((0,), dtype=torch.long, device=device),
            torch.empty((2, 0), dtype=torch.long, device=device),
        )
    edge_mask = torch.isin(edge_index_global[0], src_node_ids.to(device=edge_index_global.device)) & torch.isin(
        edge_index_global[1], dst_node_ids.to(device=edge_index_global.device)
    )
    return edge_ids_global[edge_mask], edge_index_global[:, edge_mask]


def _build_stitched_homo_link_records(graph, records, stitched_graph: Graph) -> list[LinkPredictionRecord]:
    seed_positions = {int(node_id): index for index, node_id in enumerate(stitched_graph.n_id.tolist())}
    sampled_records = []
    for record in records:
        src_global = int(_graph_node_global_ids(graph, torch.tensor([record.src_index]), node_type="node").item())
        dst_global = int(_graph_node_global_ids(graph, torch.tensor([record.dst_index]), node_type="node").item())
        sampled_records.append(
            LinkPredictionRecord(
                graph=stitched_graph,
                src_index=seed_positions[src_global],
                dst_index=seed_positions[dst_global],
                label=int(record.label),
                metadata=dict(record.metadata),
                sample_id=record.sample_id,
                exclude_seed_edge=bool(record.exclude_seed_edge),
                hard_negative_dst=record.hard_negative_dst,
                candidate_dst=record.candidate_dst,
                edge_type=record.edge_type,
                reverse_edge_type=record.reverse_edge_type,
                query_id=record.query_id,
                filter_ranking=bool(record.filter_ranking),
            )
        )
    return sampled_records


def _build_stitched_hetero_link_records(graph, records, stitched_graph: Graph) -> list[LinkPredictionRecord]:
    seed_positions_by_type = {
        node_type: {
            int(node_id): index
            for index, node_id in enumerate(stitched_graph.nodes[node_type].data["n_id"].tolist())
        }
        for node_type in stitched_graph.nodes
    }
    sampled_records = []
    for record in records:
        edge_type = _resolve_link_record_edge_type(record)
        src_type, _, dst_type = edge_type
        src_global = int(
            _graph_node_global_ids(
                graph,
                torch.tensor([record.src_index], dtype=torch.long),
                node_type=src_type,
            ).item()
        )
        dst_global = int(
            _graph_node_global_ids(
                graph,
                torch.tensor([record.dst_index], dtype=torch.long),
                node_type=dst_type,
            ).item()
        )
        sampled_records.append(
            LinkPredictionRecord(
                graph=stitched_graph,
                src_index=seed_positions_by_type[src_type][src_global],
                dst_index=seed_positions_by_type[dst_type][dst_global],
                label=int(record.label),
                metadata=dict(record.metadata),
                sample_id=record.sample_id,
                exclude_seed_edge=bool(record.exclude_seed_edge),
                hard_negative_dst=record.hard_negative_dst,
                candidate_dst=record.candidate_dst,
                edge_type=edge_type,
                reverse_edge_type=record.reverse_edge_type,
                query_id=record.query_id,
                filter_ranking=bool(record.filter_ranking),
            )
        )
    return sampled_records


@dataclass(slots=True)
class MaterializationContext:
    request: Any
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    graph: Any | None = None
    feature_store: Any | None = None


@dataclass(slots=True)
class PlanExecutor:
    handlers: dict[str, StageHandler] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.handlers.setdefault("expand_neighbors", self._expand_neighbors)
        self.handlers.setdefault("fetch_node_features", self._fetch_node_features)
        self.handlers.setdefault("fetch_edge_features", self._fetch_edge_features)
        self.handlers.setdefault("sample_link_neighbors", self._sample_link_neighbors)
        self.handlers.setdefault("sample_temporal_neighbors", self._sample_temporal_neighbors)

    def register(self, name: str, handler: StageHandler) -> None:
        self.handlers[name] = handler

    def execute(
        self,
        plan: SamplingPlan,
        *,
        graph: Any | None = None,
        feature_store: Any | None = None,
        state: dict[str, Any] | None = None,
    ) -> MaterializationContext:
        context = MaterializationContext(
            request=plan.request,
            state=dict(state or {}),
            metadata=dict(plan.metadata),
            graph=graph,
            feature_store=_resolve_feature_store(feature_store, graph),
        )
        for stage in plan.stages:
            try:
                handler = self.handlers[stage.name]
            except KeyError as exc:
                raise KeyError(f"unknown stage handler: {stage.name}") from exc
            context = handler(stage, context)
        return context

    def _expand_neighbors(self, stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        from vgl.ops.khop import expand_neighbors

        if context.graph is None:
            raise ValueError("graph is required for neighbor expansion stages")
        request = context.request
        output_blocks = bool(stage.params.get("output_blocks", False))

        if _match_partition_shard(context.graph, context.feature_store) is not None:
            seed_local_ids = torch.as_tensor(request.node_ids, dtype=torch.long).view(-1)
            if output_blocks:
                seed_global_ids, node_ids_global, node_hops = _expand_stitched_homo_node_ids(
                    context.graph,
                    context.feature_store,
                    seed_local_ids,
                    fanouts=stage.params["num_neighbors"],
                    return_hops=True,
                )
                context.state["node_hops"] = node_hops
            else:
                seed_global_ids, node_ids_global = _expand_stitched_homo_node_ids(
                    context.graph,
                    context.feature_store,
                    seed_local_ids,
                    fanouts=stage.params["num_neighbors"],
                )
            edge_ids_global, edge_index_global = _collect_stitched_homo_edges(
                context.feature_store,
                node_ids_global,
                edge_type=context.graph._default_edge_type(),
            )
            stitched_graph = _build_stitched_homo_graph(
                context.graph,
                context.feature_store,
                node_ids_global,
                edge_ids_global,
                _relabel_stitched_edge_index(node_ids_global, edge_index_global),
            )
            context.state["sample"] = _build_stitched_node_samples(
                context,
                stitched_graph,
                seed_local_ids,
                seed_global_ids,
            )
            context.state["graph"] = stitched_graph
            _store_sampled_graph_indices(context, stitched_graph)
            context.state["node_ids_global"] = context.state["node_ids"]
            context.state["edge_ids_global"] = context.state["edge_ids"]
            return context

        if _match_hetero_partition_shard(context.graph, context.feature_store) is not None:
            seed_local_ids = torch.as_tensor(request.node_ids, dtype=torch.long).view(-1)
            if output_blocks:
                seed_global_ids, node_ids_by_type, node_hops_by_type = _expand_stitched_hetero_node_ids(
                    context.graph,
                    context.feature_store,
                    seed_local_ids,
                    seed_node_type=request.node_type,
                    fanouts=stage.params["num_neighbors"],
                    generator=getattr(stage.params.get("sampler", None), "_generator", None),
                    return_hops=True,
                )
                context.state["node_hops_by_type"] = node_hops_by_type
                context.state["node_block_edge_type"] = stage.params["node_block_edge_type"]
            else:
                seed_global_ids, node_ids_by_type = _expand_stitched_hetero_node_ids(
                    context.graph,
                    context.feature_store,
                    seed_local_ids,
                    seed_node_type=request.node_type,
                    fanouts=stage.params["num_neighbors"],
                    generator=getattr(stage.params.get("sampler", None), "_generator", None),
                )
            edge_ids_by_type, edge_index_by_type = _collect_stitched_hetero_edges(
                context.feature_store,
                node_ids_by_type,
                edge_types=tuple(context.graph.edges),
            )
            stitched_graph = _build_stitched_hetero_graph(
                context.graph,
                context.feature_store,
                node_ids_by_type,
                edge_ids_by_type,
                _relabel_stitched_edge_index_by_type(node_ids_by_type, edge_index_by_type),
            )
            context.state["sample"] = _build_stitched_hetero_node_samples(
                context,
                stitched_graph,
                seed_local_ids,
                seed_global_ids,
                node_type=request.node_type,
            )
            context.state["graph"] = stitched_graph
            _store_sampled_graph_indices(context, stitched_graph)
            context.state["node_ids_by_type_global"] = context.state["node_ids_by_type"]
            context.state["edge_ids_by_type_global"] = context.state["edge_ids_by_type"]
            return context

        expanded = expand_neighbors(
            context.graph,
            request.node_ids,
            num_neighbors=stage.params["num_neighbors"],
            node_type=stage.params.get("node_type", getattr(request, "node_type", None)),
            return_hops=output_blocks,
        )
        node_hops = None
        if output_blocks:
            expanded, node_hops = expanded
        if isinstance(expanded, dict):
            edge_ids_by_type = _induced_edge_ids_by_type(context.graph, expanded)
            context.state["node_ids_by_type"] = expanded
            context.state["edge_ids_by_type"] = edge_ids_by_type
            context.state["node_ids_by_type_global"] = {
                current_type: _graph_node_global_ids(context.graph, node_ids, node_type=current_type)
                for current_type, node_ids in expanded.items()
            }
            context.state["edge_ids_by_type_global"] = {
                edge_type: _graph_edge_global_ids(context.graph, edge_ids, edge_type=edge_type)
                for edge_type, edge_ids in edge_ids_by_type.items()
            }
            if node_hops is not None:
                context.state["node_hops_by_type"] = node_hops
                context.state["node_block_edge_type"] = stage.params["node_block_edge_type"]
        else:
            edge_ids = _induced_edge_ids(context.graph, expanded)
            context.state["node_ids"] = expanded
            context.state["edge_ids"] = edge_ids
            context.state["node_ids_global"] = _graph_node_global_ids(context.graph, expanded, node_type="node")
            context.state["edge_ids_global"] = _graph_edge_global_ids(
                context.graph,
                edge_ids,
                edge_type=context.graph._default_edge_type(),
            )
            if node_hops is not None:
                context.state["node_hops"] = node_hops
        return context

    def _fetch_node_features(self, stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        return self._fetch_features(stage, context, entity_kind="node", type_key="node_type")

    def _fetch_edge_features(self, stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        return self._fetch_features(stage, context, entity_kind="edge", type_key="edge_type")

    @staticmethod
    def _sample_link_neighbors(stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        sampler = stage.params["sampler"]
        records = list(stage.params["records"])
        graph = records[0].graph
        output_blocks = bool(stage.params.get("output_blocks", False))
        stitched_homo_partition = _match_partition_shard(graph, context.feature_store)
        stitched_hetero_partition = None
        if stitched_homo_partition is None:
            stitched_hetero_partition = _match_hetero_partition_shard(graph, context.feature_store)
        if stitched_homo_partition is not None:
            seed_local_ids = torch.tensor(
                [int(node) for record in records for node in (record.src_index, record.dst_index)],
                dtype=torch.long,
            )
            seed_global_ids = _graph_node_global_ids(graph, seed_local_ids, node_type="node")
            link_node_hops = None
            if output_blocks:
                node_ids_global, link_node_hops = _expand_stitched_homo_global_node_ids(
                    context.feature_store,
                    seed_global_ids,
                    fanouts=sampler.num_neighbors,
                    edge_type=graph._default_edge_type(),
                    generator=getattr(sampler, "_generator", None),
                    return_hops=True,
                )
            else:
                node_ids_global = _expand_stitched_homo_global_node_ids(
                    context.feature_store,
                    seed_global_ids,
                    fanouts=sampler.num_neighbors,
                    edge_type=graph._default_edge_type(),
                    generator=getattr(sampler, "_generator", None),
                )
            edge_ids_global, edge_index_global = _collect_stitched_homo_edges(
                context.feature_store,
                node_ids_global,
                edge_type=graph._default_edge_type(),
            )
            stitched_graph = _build_stitched_homo_graph(
                graph,
                context.feature_store,
                node_ids_global,
                edge_ids_global,
                _relabel_stitched_edge_index(node_ids_global, edge_index_global),
            )
            sampled_records = _build_stitched_homo_link_records(graph, records, stitched_graph)
            sampled = sampled_records if bool(stage.params["is_sequence"]) else sampled_records[0]
            if link_node_hops is not None:
                context.state["link_node_ids_local"] = stitched_graph.n_id
                context.state["link_node_hops"] = link_node_hops
        elif stitched_hetero_partition is not None:
            seed_global_ids_by_type = _stitched_hetero_link_seed_global_ids(graph, records)
            if output_blocks:
                node_ids_by_type, link_node_hops_by_type = _expand_stitched_hetero_global_node_ids(
                    context.feature_store,
                    seed_global_ids_by_type,
                    edge_types=tuple(graph.edges),
                    fanouts=sampler.num_neighbors,
                    generator=getattr(sampler, "_generator", None),
                    return_hops=True,
                )
                context.state["link_node_hops_by_type"] = link_node_hops_by_type
                context.state["link_block_edge_type"] = _single_link_record_edge_type(records)
            else:
                node_ids_by_type = _expand_stitched_hetero_global_node_ids(
                    context.feature_store,
                    seed_global_ids_by_type,
                    edge_types=tuple(graph.edges),
                    fanouts=sampler.num_neighbors,
                    generator=getattr(sampler, "_generator", None),
                )
            edge_ids_by_type, edge_index_by_type = _collect_stitched_hetero_edges(
                context.feature_store,
                node_ids_by_type,
                edge_types=tuple(graph.edges),
            )
            stitched_graph = _build_stitched_hetero_graph(
                graph,
                context.feature_store,
                node_ids_by_type,
                edge_ids_by_type,
                _relabel_stitched_edge_index_by_type(node_ids_by_type, edge_index_by_type),
            )
            sampled_records = _build_stitched_hetero_link_records(graph, records, stitched_graph)
            sampled = sampled_records if bool(stage.params["is_sequence"]) else sampled_records[0]
        else:
            if output_blocks:
                sampled, block_state = sampler._sample_from_seed_records(
                    records,
                    is_sequence=bool(stage.params["is_sequence"]),
                    return_hops=True,
                )
                context.state.update(block_state)
            else:
                sampled = sampler._sample_from_seed_records(records, is_sequence=bool(stage.params["is_sequence"]))
        if isinstance(sampled, (list, tuple)):
            sampled_records = list(sampled)
            context.state["records"] = sampled_records
            sampled_graph = sampled_records[0].graph
        else:
            context.state["record"] = sampled
            sampled_graph = sampled.graph
        _store_sampled_graph_indices(context, sampled_graph)
        if stitched_homo_partition is not None:
            context.state["node_ids_global"] = context.state["node_ids"]
            context.state["edge_ids_global"] = context.state["edge_ids"]
        elif stitched_hetero_partition is not None:
            context.state["node_ids_by_type_global"] = context.state["node_ids_by_type"]
            context.state["edge_ids_by_type_global"] = context.state["edge_ids_by_type"]
        return context

    @staticmethod
    def _sample_temporal_neighbors(stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        sampler = stage.params["sampler"]
        seed_record = stage.params["record"]
        graph = seed_record.graph
        stitched_homo_partition = _match_temporal_partition_shard(graph, context.feature_store)
        stitched_hetero_temporal_partition = None
        if stitched_homo_partition is None:
            stitched_hetero_temporal_partition = _match_hetero_temporal_partition_shard(graph, context.feature_store)
        if stitched_homo_partition is not None:
            edge_type = _resolve_temporal_record_edge_type(seed_record)
            seed_local_ids = torch.tensor([int(seed_record.src_index), int(seed_record.dst_index)], dtype=torch.long)
            seed_global_ids = _graph_node_global_ids(graph, seed_local_ids, node_type="node")
            history_edge_ids_global, history_edge_index_global = _build_stitched_homo_temporal_history(
                graph,
                context.feature_store,
                sampler,
                seed_record,
                edge_type=edge_type,
            )
            node_ids_global = _expand_stitched_homo_temporal_node_ids(
                seed_global_ids,
                history_edge_index_global,
                fanouts=sampler.num_neighbors,
                generator=getattr(sampler, "_generator", None),
            )
            edge_ids_global, edge_index_global = _induce_stitched_homo_temporal_edges(
                node_ids_global,
                history_edge_ids_global,
                history_edge_index_global,
            )
            stitched_graph = _build_stitched_homo_temporal_graph(
                graph,
                context.feature_store,
                node_ids_global,
                edge_ids_global,
                _relabel_stitched_edge_index(node_ids_global, edge_index_global),
                edge_type=edge_type,
            )
            record = _build_stitched_homo_temporal_record(graph, seed_record, stitched_graph)
        elif stitched_hetero_temporal_partition is not None:
            edge_type = _resolve_temporal_record_edge_type(seed_record)
            src_type, _, dst_type = edge_type
            seed_global_ids_by_type = _stitched_hetero_temporal_seed_global_ids(
                graph,
                seed_record,
                edge_type=edge_type,
            )
            history_edge_ids_global, history_edge_index_global = _build_stitched_hetero_temporal_history(
                graph,
                context.feature_store,
                sampler,
                seed_record,
                edge_type=edge_type,
            )
            node_ids_by_type = _expand_stitched_hetero_temporal_node_ids(
                seed_global_ids_by_type,
                history_edge_index_global,
                src_type=src_type,
                dst_type=dst_type,
                fanouts=sampler.num_neighbors,
                generator=getattr(sampler, "_generator", None),
            )
            edge_ids_global, edge_index_global = _induce_stitched_hetero_temporal_edges(
                node_ids_by_type,
                history_edge_ids_global,
                history_edge_index_global,
                src_type=src_type,
                dst_type=dst_type,
            )
            edge_index = _relabel_stitched_edge_index_by_type(
                node_ids_by_type,
                {edge_type: edge_index_global},
            )[edge_type]
            stitched_graph = _build_stitched_hetero_temporal_graph(
                graph,
                context.feature_store,
                node_ids_by_type,
                edge_ids_global,
                edge_index,
                edge_type=edge_type,
            )
            record = _build_stitched_hetero_temporal_record(graph, seed_record, stitched_graph)
        else:
            record = sampler._sample_event(seed_record)
        context.state["record"] = record
        _store_sampled_graph_indices(context, record.graph)
        if stitched_homo_partition is not None:
            context.state["node_ids_global"] = context.state["node_ids"]
            context.state["edge_ids_global"] = context.state["edge_ids"]
        elif stitched_hetero_temporal_partition is not None:
            if set(record.graph.nodes) == {"node"} and len(record.graph.edges) == 1:
                context.state["node_ids_global"] = context.state["node_ids"]
                context.state["edge_ids_global"] = context.state["edge_ids"]
            else:
                context.state["node_ids_by_type_global"] = context.state["node_ids_by_type"]
                context.state["edge_ids_by_type_global"] = context.state["edge_ids_by_type"]
        return context

    @staticmethod
    def _fetch_features(
        stage: PlanStage,
        context: MaterializationContext,
        *,
        entity_kind: str,
        type_key: str,
    ) -> MaterializationContext:
        source = context.feature_store
        if source is None:
            raise ValueError("feature_store is required for feature fetch stages")
        output_key = stage.params["output_key"]
        type_name = stage.params[type_key]
        if entity_kind == "edge":
            type_name = tuple(type_name)

        direct_fetch = getattr(source, "fetch", None)
        node_fetch = getattr(source, "fetch_node_features", None)
        edge_fetch = getattr(source, "fetch_edge_features", None)
        routed_source = not callable(direct_fetch) and (
            (entity_kind == "node" and callable(node_fetch))
            or (entity_kind == "edge" and callable(edge_fetch))
        )
        requested_index = _resolve_state_index(context, stage.params["index_key"], type_name=type_name)
        fetch_index = _resolve_fetch_index(stage, context, type_name=type_name, routed_source=routed_source)

        fetched = {}
        for feature_name in stage.params["feature_names"]:
            key = (entity_kind, type_name, feature_name)
            if callable(direct_fetch):
                tensor_slice = direct_fetch(key, fetch_index)
            elif entity_kind == "node" and callable(node_fetch):
                tensor_slice = node_fetch(key, fetch_index)
            elif entity_kind == "edge" and callable(edge_fetch):
                try:
                    tensor_slice = edge_fetch(key, fetch_index)
                except KeyError:
                    if routed_source:
                        tensor_slice = _fallback_routed_edge_tensor_slice(source, key, fetch_index)
                    else:
                        raise
            else:
                raise TypeError(
                    f"feature_store does not support {entity_kind} feature fetch for key {key!r}"
                )
            if routed_source and isinstance(tensor_slice, TensorSlice):
                tensor_slice = TensorSlice(
                    index=torch.as_tensor(requested_index, dtype=torch.long, device=tensor_slice.index.device).view(-1),
                    values=tensor_slice.values,
                )
            fetched[feature_name] = tensor_slice
        context.state[output_key] = fetched
        _record_materialized_features(context, entity_kind=entity_kind, type_name=type_name, fetched=fetched)
        return context
