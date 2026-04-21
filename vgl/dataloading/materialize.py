from collections.abc import Mapping
from dataclasses import replace
from typing import cast

import torch

from vgl.dataloading.executor import _as_python_int
from vgl.dataloading.executor import MaterializationContext, _lookup_positions
from vgl.dataloading.records import LinkPredictionRecord, SampleRecord, TemporalEventRecord
from vgl.graph.batch import (
    GraphBatch,
    LinkPredictionBatch,
    NodeBatch,
    TemporalEventBatch,
    _resolve_link_edge_type,
    _resolve_link_reverse_edge_type,
    _without_supervision_edges,
    _without_supervision_edges_for_type,
)
from vgl.graph.graph import Graph
from vgl.ops.block import to_block, to_hetero_block
from vgl.ops.subgraph import _membership_mask, _positions_for_endpoint_values, _relabel_bipartite_edge_index


def _cache_key(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return ("tensor", str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes())
    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (key, _cache_key(inner_value))
                for key, inner_value in sorted(value.items(), key=lambda item: repr(item[0]))
            ),
        )
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_cache_key(item) for item in value))
    if hasattr(value, "index") and hasattr(value, "values"):
        return ("slice", _cache_key(value.index), _cache_key(value.values))
    return ("object", id(value))


def _resolved_seed_metadata_values(seed_value, fallback_seeds: torch.Tensor) -> list[int]:
    fallback = [_as_python_int(seed) for seed in torch.as_tensor(fallback_seeds, dtype=torch.long).view(-1)]
    if seed_value is None:
        return fallback
    if isinstance(seed_value, torch.Tensor):
        values = [_as_python_int(seed) for seed in seed_value.view(-1)]
    elif isinstance(seed_value, (list, tuple)):
        values = [_as_python_int(seed) for seed in seed_value]
    else:
        values = [_as_python_int(seed_value)]
    if len(values) == len(fallback):
        return values
    if len(values) == 1 and len(fallback) == 1:
        return values
    raise ValueError("seed metadata must align with the number of requested seeds")


def _align_tensor_slice(index: torch.Tensor, tensor_slice) -> torch.Tensor:
    index = torch.as_tensor(index, dtype=torch.long).view(-1)
    slice_index = tensor_slice.index.to(dtype=torch.long).view(-1)
    if index.numel() == 0:
        return tensor_slice.values.new_empty((0,) + tuple(tensor_slice.values.shape[1:]))
    if torch.equal(index, slice_index):
        return tensor_slice.values
    order_index = _lookup_positions(slice_index, index, entity_name="feature slice")
    return tensor_slice.values[order_index.to(device=tensor_slice.values.device)]


def _subgraph(
    graph: Graph,
    node_ids: torch.Tensor,
    *,
    fetched_node_features: dict[str, dict[str, object]] | None = None,
    fetched_edge_features: dict[object, dict[str, object]] | None = None,
):
    node_ids = node_ids.to(dtype=torch.long)
    num_nodes = graph._node_count("node")
    edge_index = graph.edge_index
    edge_store = graph.edges[graph._default_edge_type()]
    node_ids_device = node_ids.to(device=edge_index.device)
    candidate_edge_ids = _positions_for_endpoint_values(edge_store, node_ids_device, endpoint=0)
    edge_ids = candidate_edge_ids[
        _membership_mask(edge_index[1, candidate_edge_ids], node_ids_device)
    ]
    if edge_ids.numel() == 0:
        subgraph_edge_index = edge_index[:, :0]
    else:
        subgraph_edge_index = _relabel_bipartite_edge_index(
            edge_index[:, edge_ids],
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
    for feature_name, tensor_slice in (fetched_node_features or {}).get("node", {}).items():
        node_data[feature_name] = _align_tensor_slice(node_ids, tensor_slice)

    edge_type = graph._default_edge_type()
    edge_count = int(edge_store.edge_index.size(1))
    edge_data = {}
    for key, value in edge_store.data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[edge_ids]
        else:
            edge_data[key] = value
    if "e_id" not in edge_data:
        edge_data["e_id"] = edge_ids
    for feature_name, tensor_slice in (fetched_edge_features or {}).get(edge_type, {}).items():
        edge_data[feature_name] = _align_tensor_slice(edge_ids, tensor_slice)
    return Graph.homo(edge_index=subgraph_edge_index, edge_data=edge_data, **node_data), node_ids


def _hetero_subgraph(
    graph: Graph,
    node_ids_by_type: dict[str, torch.Tensor],
    *,
    fetched_node_features: dict[str, dict[str, object]] | None = None,
    fetched_edge_features: dict[object, dict[str, object]] | None = None,
):
    nodes = {}
    for node_type, store in graph.nodes.items():
        node_ids = node_ids_by_type[node_type].to(dtype=torch.long)
        num_nodes = graph._node_count(node_type)

        node_data = {}
        for key, value in store.data.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == num_nodes:
                node_data[key] = value[node_ids]
            else:
                node_data[key] = value
        if "n_id" not in node_data:
            node_data["n_id"] = node_ids
        for feature_name, tensor_slice in (fetched_node_features or {}).get(node_type, {}).items():
            node_data[feature_name] = _align_tensor_slice(node_ids, tensor_slice)
        nodes[node_type] = node_data

    edges = {}
    for edge_type, edge_store in graph.edges.items():
        src_type, _, dst_type = edge_type
        edge_index = edge_store.edge_index
        src_node_ids = node_ids_by_type[src_type].to(dtype=torch.long, device=edge_store.edge_index.device)
        dst_node_ids = node_ids_by_type[dst_type].to(dtype=torch.long, device=edge_store.edge_index.device)
        candidate_edge_ids = _positions_for_endpoint_values(edge_store, src_node_ids, endpoint=0)
        edge_ids = candidate_edge_ids[
            _membership_mask(edge_store.edge_index[1, candidate_edge_ids], dst_node_ids)
        ]
        if edge_ids.numel() == 0:
            subgraph_edge_index = edge_store.edge_index[:, :0]
        else:
            subgraph_edge_index = _relabel_bipartite_edge_index(
                edge_store.edge_index[:, edge_ids],
                node_ids_by_type[src_type],
                node_ids_by_type[dst_type],
            )
        edge_count = int(edge_index.size(1))
        edge_data = {"edge_index": subgraph_edge_index}
        for key, value in edge_store.data.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_data[key] = value[edge_ids]
            else:
                edge_data[key] = value
        if "e_id" not in edge_data:
            edge_data["e_id"] = edge_ids
        for feature_name, tensor_slice in (fetched_edge_features or {}).get(edge_type, {}).items():
            edge_data[feature_name] = _align_tensor_slice(edge_ids, tensor_slice)
        edges[edge_type] = edge_data
    return Graph.hetero(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr), {
        node_type: node_ids_by_type[node_type].to(dtype=torch.long)
        for node_type in graph.nodes
    }


def _node_store_shape(store) -> tuple[int, torch.device]:
    for value in store.data.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.size(0)), value.device
    raise ValueError(f"cannot infer node shape for node type {store.type_name!r}")


def _node_data_device(node_data: Mapping[str, object]) -> torch.device:
    for value in node_data.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _store_device(store) -> torch.device:
    for value in store.data.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _graph_with_materialized_features(
    graph: Graph,
    *,
    fetched_node_features: dict[str, dict[str, object]] | None = None,
    fetched_edge_features: dict[object, dict[str, object]] | None = None,
) -> Graph:
    if not fetched_node_features and not fetched_edge_features:
        return graph

    if graph.schema.time_attr is None and set(graph.nodes) == {"node"} and len(graph.edges) == 1:
        node_store = graph.nodes["node"]
        node_data = dict(node_store.data)
        node_ids = node_data.get("n_id")
        if node_ids is None:
            node_count = graph._node_count("node")
            node_ids = torch.arange(node_count, dtype=torch.long, device=_node_data_device(node_data))
            node_data["n_id"] = node_ids
        for feature_name, tensor_slice in (fetched_node_features or {}).get("node", {}).items():
            node_data[feature_name] = _align_tensor_slice(node_ids, tensor_slice)

        edge_type = graph._default_edge_type()
        edge_store = graph.edges[edge_type]
        edge_data = {key: value for key, value in edge_store.data.items() if key != "edge_index"}
        edge_ids = edge_data.get("e_id")
        if edge_ids is None:
            edge_ids = torch.arange(edge_store.edge_index.size(1), dtype=torch.long, device=edge_store.edge_index.device)
            edge_data["e_id"] = edge_ids
        for feature_name, tensor_slice in (fetched_edge_features or {}).get(edge_type, {}).items():
            edge_data[feature_name] = _align_tensor_slice(edge_ids, tensor_slice)
        materialized = Graph.homo(edge_index=edge_store.edge_index, edge_data=edge_data, **node_data)
    else:
        nodes = {}
        for node_type, store in graph.nodes.items():
            node_data = dict(store.data)
            node_ids = node_data.get("n_id")
            if node_ids is None:
                node_count = graph._node_count(node_type)
                device = _node_data_device(node_data)
                node_ids = torch.arange(node_count, dtype=torch.long, device=device)
                node_data["n_id"] = node_ids
            for feature_name, tensor_slice in (fetched_node_features or {}).get(node_type, {}).items():
                node_data[feature_name] = _align_tensor_slice(node_ids, tensor_slice)
            nodes[node_type] = node_data

        edges = {}
        for edge_type, edge_store in graph.edges.items():
            edge_data = dict(edge_store.data)
            edge_ids = edge_data.get("e_id")
            if edge_ids is None:
                edge_ids = torch.arange(edge_store.edge_index.size(1), dtype=torch.long, device=edge_store.edge_index.device)
                edge_data["e_id"] = edge_ids
            for feature_name, tensor_slice in (fetched_edge_features or {}).get(edge_type, {}).items():
                edge_data[feature_name] = _align_tensor_slice(edge_ids, tensor_slice)
            edges[edge_type] = edge_data
        if graph.schema.time_attr is not None:
            materialized = Graph.temporal(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr)
        else:
            materialized = Graph.hetero(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr)

    materialized.feature_store = graph.feature_store
    return materialized


def _build_homo_blocks_from_local_ids(subgraph: Graph, node_ids_local: torch.Tensor, node_hops: list[torch.Tensor]):
    node_ids_local = torch.as_tensor(node_ids_local, dtype=torch.long).view(-1)
    subgraph_hops = []
    for hop_node_ids in node_hops:
        hop_node_ids = torch.as_tensor(hop_node_ids, dtype=torch.long).view(-1)
        subgraph_hops.append(
            _lookup_positions(node_ids_local, hop_node_ids, entity_name="block node").to(device=node_ids_local.device)
        )
    return [to_block(subgraph, dst_nodes) for dst_nodes in subgraph_hops[-2::-1]]


def _build_hetero_blocks_from_local_ids(
    subgraph: Graph,
    node_hops_by_type: list[dict[str, torch.Tensor]],
    *,
    edge_type,
):
    dst_type = edge_type[2]
    dst_node_ids = subgraph.nodes[dst_type].data.get("n_id")
    if dst_node_ids is None:
        node_count = subgraph._node_count(dst_type)
        device = _store_device(subgraph.nodes[dst_type])
        dst_node_ids = torch.arange(node_count, dtype=torch.long, device=device)
    dst_node_ids = torch.as_tensor(dst_node_ids, dtype=torch.long).view(-1)
    subgraph_hops = []
    for hop_nodes in node_hops_by_type:
        hop_node_ids = torch.as_tensor(hop_nodes.get(dst_type, ()), dtype=torch.long).view(-1)
        subgraph_hops.append(
            _lookup_positions(dst_node_ids, hop_node_ids, entity_name=f"{dst_type} block node").to(
                device=dst_node_ids.device
            )
        )
    return [to_block(subgraph, dst_nodes, edge_type=edge_type) for dst_nodes in subgraph_hops[-2::-1]]


def _build_full_hetero_blocks_from_local_ids(
    subgraph: Graph,
    node_hops_by_type: list[dict[str, torch.Tensor]],
):
    devices_by_type = {}
    for node_type, store in subgraph.nodes.items():
        node_ids = store.data.get("n_id")
        if node_ids is None:
            node_count = subgraph._node_count(node_type)
            device = _store_device(store)
            node_ids = torch.arange(node_count, dtype=torch.long, device=device)
        node_ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
        devices_by_type[node_type] = node_ids

    subgraph_hops = []
    for hop_nodes in node_hops_by_type:
        dst_nodes_by_type = {}
        for node_type in subgraph.schema.node_types:
            hop_node_ids = torch.as_tensor(hop_nodes.get(node_type, ()), dtype=torch.long).view(-1)
            node_ids = devices_by_type[node_type]
            dst_nodes_by_type[node_type] = _lookup_positions(
                node_ids,
                hop_node_ids,
                entity_name=f"{node_type} block node",
            ).to(
                device=node_ids.device,
            )
        subgraph_hops.append(dst_nodes_by_type)
    return [
        to_hetero_block(subgraph, dst_nodes_by_type, edge_types=tuple(subgraph.edges))
        for dst_nodes_by_type in subgraph_hops[-2::-1]
    ]


def _link_message_passing_graph(graph: Graph, records: list[LinkPredictionRecord]) -> Graph:
    supervision_edges_by_type: dict[tuple[str, str, str], set[tuple[int, int]]] = {}
    reverse_supervision_edges: dict[tuple[str, str, str], set[tuple[int, int]]] = {}
    for record in records:
        if _as_python_int(record.label) != 1:
            continue
        if not (bool(getattr(record, "exclude_seed_edge", False)) or bool(record.metadata.get("exclude_seed_edges", False))):
            continue
        edge_type = getattr(record, "edge_type", None) or record.metadata.get("edge_type")
        if edge_type is None:
            edge_type = graph._default_edge_type()
        edge_type = tuple(edge_type)
        supervision_edges_by_type.setdefault(edge_type, set()).add(
            (_as_python_int(record.src_index), _as_python_int(record.dst_index))
        )
        reverse_edge_type = getattr(record, "reverse_edge_type", None)
        if reverse_edge_type is None:
            reverse_edge_type = record.metadata.get("reverse_edge_type")
        if reverse_edge_type is not None:
            reverse_edge_type = tuple(reverse_edge_type)
            reverse_supervision_edges.setdefault(reverse_edge_type, set()).add(
                (_as_python_int(record.dst_index), _as_python_int(record.src_index))
            )
    if set(graph.nodes) == {"node"} and len(graph.edges) == 1:
        supervision_edges = supervision_edges_by_type.get(graph._default_edge_type(), set())
        if reverse_supervision_edges:
            supervision_edges = set(supervision_edges)
            for reverse_edges in reverse_supervision_edges.values():
                supervision_edges.update(reverse_edges)
        return _without_supervision_edges(graph, supervision_edges)
    message_passing_graph = graph
    for edge_type, supervision_edges in supervision_edges_by_type.items():
        message_passing_graph = _without_supervision_edges_for_type(message_passing_graph, edge_type, supervision_edges)
    for edge_type, supervision_edges in reverse_supervision_edges.items():
        message_passing_graph = _without_supervision_edges_for_type(message_passing_graph, edge_type, supervision_edges)
    return message_passing_graph


def _materialize_record_payload(
    context: MaterializationContext,
    payload,
    *,
    graph_cache: dict[object, Graph] | None = None,
    node_blocks_cache: dict[object, list[object]] | None = None,
    link_blocks_cache: dict[object, list[object]] | None = None,
):
    fetched_node_features = context.state.get("_materialized_node_features")
    fetched_edge_features = context.state.get("_materialized_edge_features")
    needs_homo_node_blocks = "node_hops" in context.state
    needs_hetero_node_blocks = "node_hops_by_type" in context.state
    needs_node_blocks = needs_homo_node_blocks or needs_hetero_node_blocks
    needs_homo_link_blocks = "link_node_hops" in context.state and "link_node_ids_local" in context.state
    needs_hetero_link_blocks = "link_node_hops_by_type" in context.state
    needs_link_blocks = needs_homo_link_blocks or needs_hetero_link_blocks
    if not fetched_node_features and not fetched_edge_features and not needs_node_blocks and not needs_link_blocks:
        return payload

    graph_cache = {} if graph_cache is None else graph_cache
    node_blocks_cache = {} if node_blocks_cache is None else node_blocks_cache
    link_blocks_cache = {} if link_blocks_cache is None else link_blocks_cache

    def _materialized_graph(graph):
        graph_id = (
            id(graph),
            _cache_key(fetched_node_features),
            _cache_key(fetched_edge_features),
        )
        materialized = graph_cache.get(graph_id)
        if materialized is None:
            materialized = _graph_with_materialized_features(
                graph,
                fetched_node_features=fetched_node_features,
                fetched_edge_features=fetched_edge_features,
            )
            graph_cache[graph_id] = materialized
        return materialized

    def _materialized_node_blocks(graph: Graph):
        cache_key = (
            id(graph),
            context.state.get("node_block_edge_type"),
            _cache_key(context.state.get("node_hops")),
            _cache_key(context.state.get("node_hops_by_type")),
        )
        blocks = node_blocks_cache.get(cache_key)
        if blocks is None:
            if needs_homo_node_blocks:
                node_ids = graph.nodes["node"].data.get("n_id")
                if node_ids is None:
                    node_ids = torch.arange(
                        graph._node_count("node"),
                        dtype=torch.long,
                        device=_node_data_device(graph.nodes["node"].data),
                    )
                blocks = _build_homo_blocks_from_local_ids(graph, node_ids, context.state["node_hops"])
            elif context.state.get("node_block_edge_type") is not None:
                blocks = _build_hetero_blocks_from_local_ids(
                    graph,
                    context.state["node_hops_by_type"],
                    edge_type=context.state["node_block_edge_type"],
                )
            else:
                blocks = _build_full_hetero_blocks_from_local_ids(graph, context.state["node_hops_by_type"])
            node_blocks_cache[cache_key] = blocks
        return blocks

    def _materialized_link_blocks(records: list[LinkPredictionRecord], graph: Graph):
        supervision_signature = []
        for record in records:
            if _as_python_int(record.label) != 1:
                continue
            if not (
                bool(getattr(record, "exclude_seed_edge", False))
                or bool(record.metadata.get("exclude_seed_edges", False))
            ):
                continue
            supervision_signature.append(
                (
                    _resolve_link_edge_type(record),
                    _as_python_int(record.src_index),
                    _as_python_int(record.dst_index),
                    _resolve_link_reverse_edge_type(record),
                )
            )
        cache_key = (
            id(graph),
            context.state.get("link_block_edge_type"),
            tuple(supervision_signature),
            _cache_key(context.state.get("link_node_ids_local")),
            _cache_key(context.state.get("link_node_hops")),
            _cache_key(context.state.get("link_node_hops_by_type")),
        )
        blocks = link_blocks_cache.get(cache_key)
        if blocks is None:
            message_passing_graph = _link_message_passing_graph(graph, records)
            if needs_homo_link_blocks:
                blocks = _build_homo_blocks_from_local_ids(
                    message_passing_graph,
                    context.state["link_node_ids_local"],
                    context.state["link_node_hops"],
                )
            elif context.state.get("link_block_edge_type") is not None:
                blocks = _build_hetero_blocks_from_local_ids(
                    message_passing_graph,
                    context.state["link_node_hops_by_type"],
                    edge_type=context.state["link_block_edge_type"],
                )
            else:
                blocks = _build_full_hetero_blocks_from_local_ids(
                    message_passing_graph,
                    context.state["link_node_hops_by_type"],
                )
            link_blocks_cache[cache_key] = blocks
        return blocks

    def _replace_sample_payload(samples: list[SampleRecord]):
        graph = _materialized_graph(samples[0].graph)
        blocks = _materialized_node_blocks(graph) if needs_node_blocks else None
        return [replace(sample, graph=graph, blocks=blocks) for sample in samples]

    def _replace_link_payload(records: list[LinkPredictionRecord]):
        graph = _materialized_graph(records[0].graph)
        blocks = _materialized_link_blocks(records, graph) if needs_link_blocks else None
        return [replace(record, graph=graph, blocks=blocks) for record in records]

    if isinstance(payload, list):
        if payload and all(isinstance(record, LinkPredictionRecord) for record in payload):
            return _replace_link_payload(payload)
        if payload and all(isinstance(sample, SampleRecord) for sample in payload):
            return _replace_sample_payload(payload)
        return [replace(record, graph=_materialized_graph(record.graph)) for record in payload]
    if isinstance(payload, LinkPredictionRecord):
        return _replace_link_payload([payload])[0]
    if isinstance(payload, SampleRecord):
        return _replace_sample_payload([payload])[0]
    return replace(payload, graph=_materialized_graph(payload.graph))


def _node_context_to_sample(
    context: MaterializationContext,
    *,
    graph_cache: dict[object, Graph] | None = None,
    subgraph_cache: dict[object, tuple[Graph, object]] | None = None,
    node_blocks_cache: dict[object, list[object]] | None = None,
) -> SampleRecord | list[SampleRecord]:
    if "sample" in context.state:
        return _materialize_record_payload(
            context,
            context.state["sample"],
            graph_cache=graph_cache,
            node_blocks_cache=node_blocks_cache,
        )
    if context.graph is None:
        raise ValueError("node context requires graph for materialization")
    subgraph_cache = {} if subgraph_cache is None else subgraph_cache
    node_blocks_cache = {} if node_blocks_cache is None else node_blocks_cache
    request = context.request
    metadata = dict(getattr(request, "metadata", {}))
    sample_id = context.metadata.get("sample_id", metadata.get("sample_id"))
    source_graph_id = context.metadata.get("source_graph_id", metadata.get("source_graph_id"))
    seeds = torch.as_tensor(request.node_ids, dtype=torch.long).view(-1)
    if seeds.numel() == 0:
        raise ValueError("node context requires at least one seed for materialization")

    fetched_node_features = context.state.get("_materialized_node_features")
    fetched_edge_features = context.state.get("_materialized_edge_features")

    if "node_ids" in context.state:
        cache_key: object = (
            "homo",
            id(context.graph),
            _cache_key(context.state["node_ids"]),
            _cache_key(fetched_node_features),
            _cache_key(fetched_edge_features),
        )
        cached = subgraph_cache.get(cache_key)
        if cached is None:
            cached = _subgraph(
                context.graph,
                context.state["node_ids"],
                fetched_node_features=fetched_node_features,
                fetched_edge_features=fetched_edge_features,
            )
            subgraph_cache[cache_key] = cached
        subgraph, node_mapping_obj = cached
        node_mapping = cast(torch.Tensor, node_mapping_obj)
        subgraph_seeds = _lookup_positions(
            node_mapping,
            seeds.to(device=node_mapping.device),
            entity_name="seed node",
        )
        node_type = "node"
    elif "node_ids_by_type" in context.state:
        node_type = request.node_type
        cache_key = (
            "hetero",
            id(context.graph),
            _cache_key(context.state["node_ids_by_type"]),
            _cache_key(fetched_node_features),
            _cache_key(fetched_edge_features),
        )
        cached = subgraph_cache.get(cache_key)
        if cached is None:
            cached = _hetero_subgraph(
                context.graph,
                context.state["node_ids_by_type"],
                fetched_node_features=fetched_node_features,
                fetched_edge_features=fetched_edge_features,
            )
            subgraph_cache[cache_key] = cached
        subgraph, node_mapping_obj = cached
        node_mapping = cast(dict[str, torch.Tensor], node_mapping_obj)
        subgraph_seeds = _lookup_positions(
            node_mapping[node_type],
            seeds.to(device=node_mapping[node_type].device),
            entity_name=f"{node_type} seed node",
        )
    else:
        raise ValueError("node context must contain expanded node ids for materialization")

    blocks = None
    if "node_hops" in context.state:
        if node_type != "node":
            raise ValueError("node block materialization currently supports homogeneous graphs only")
        node_blocks_cache_key: object = ("homo", id(subgraph), _cache_key(context.state["node_hops"]))
        blocks = node_blocks_cache.get(node_blocks_cache_key)
        if blocks is None:
            blocks = _build_homo_blocks_from_local_ids(subgraph, node_mapping, context.state["node_hops"])
            node_blocks_cache[node_blocks_cache_key] = blocks
    elif "node_hops_by_type" in context.state:
        if context.state.get("node_block_edge_type") is not None:
            cache_key = (
                "hetero-rel",
                id(subgraph),
                _cache_key(context.state["node_hops_by_type"]),
                context.state["node_block_edge_type"],
            )
            blocks = node_blocks_cache.get(cache_key)
            if blocks is None:
                blocks = _build_hetero_blocks_from_local_ids(
                    subgraph,
                    context.state["node_hops_by_type"],
                    edge_type=context.state["node_block_edge_type"],
                )
                node_blocks_cache[cache_key] = blocks
        else:
            cache_key = ("hetero-full", id(subgraph), _cache_key(context.state["node_hops_by_type"]))
            blocks = node_blocks_cache.get(cache_key)
            if blocks is None:
                blocks = _build_full_hetero_blocks_from_local_ids(
                    subgraph,
                    context.state["node_hops_by_type"],
                )
                node_blocks_cache[cache_key] = blocks

    samples = []
    resolved_seed_values = _resolved_seed_metadata_values(metadata.get("seed"), seeds)
    for index in range(seeds.numel()):
        sample_metadata = dict(metadata)
        sample_metadata["seed"] = resolved_seed_values[index]
        if node_type != "node":
            sample_metadata.setdefault("node_type", node_type)
        samples.append(
            SampleRecord(
                graph=subgraph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                subgraph_seed=_as_python_int(subgraph_seeds[index]),
                blocks=blocks,
            )
        )
    if len(samples) == 1:
        return samples[0]
    return samples


def _graph_context_to_sample(context: MaterializationContext) -> SampleRecord:
    if "sample" in context.state:
        return context.state["sample"]
    graph = context.state.get("graph", context.graph)
    if graph is None:
        raise ValueError("graph context requires graph for materialization")
    request = context.request
    metadata = dict(getattr(request, "metadata", {}))
    sample_id = context.metadata.get("sample_id", metadata.get("sample_id"))
    source_graph_id = context.metadata.get("source_graph_id", metadata.get("source_graph_id"))
    if sample_id is not None:
        metadata.setdefault("sample_id", sample_id)
    if source_graph_id is not None:
        metadata.setdefault("source_graph_id", source_graph_id)
    return SampleRecord(
        graph=graph,
        metadata=metadata,
        sample_id=sample_id,
        source_graph_id=source_graph_id,
    )


def _record_from_context(context: MaterializationContext):
    if "record" in context.state:
        return context.state["record"]
    if "records" in context.state:
        return context.state["records"]
    raise ValueError("record context requires 'record' or 'records' state")


def materialize_context(context: MaterializationContext):
    kind = getattr(context.request, "kind", None)
    if kind == "node":
        return _node_context_to_sample(context)
    if kind == "graph":
        return _graph_context_to_sample(context)
    if kind in {"link", "temporal"}:
        return _materialize_record_payload(context, _record_from_context(context))
    raise ValueError(f"unsupported context request kind: {kind}")


def materialize_batch(items, *, label_source=None, label_key=None):
    if not items:
        raise ValueError("materialize_batch requires at least one item")

    first = items[0]
    if isinstance(first, MaterializationContext):
        kind = getattr(first.request, "kind", None)
        if kind == "node":
            samples = []
            graph_cache: dict[object, Graph] = {}
            subgraph_cache: dict[object, tuple[Graph, object]] = {}
            node_blocks_cache: dict[object, list[object]] = {}
            for context in items:
                sample = _node_context_to_sample(
                    context,
                    graph_cache=graph_cache,
                    subgraph_cache=subgraph_cache,
                    node_blocks_cache=node_blocks_cache,
                )
                if isinstance(sample, list):
                    samples.extend(sample)
                else:
                    samples.append(sample)
            return NodeBatch.from_samples(samples)
        if kind == "graph":
            samples = [_graph_context_to_sample(context) for context in items]
            return GraphBatch.from_samples(samples, label_source=label_source, label_key=label_key)
        if kind == "link":
            records = []
            graph_cache: dict[object, Graph] = {}
            node_blocks_cache: dict[object, list[object]] = {}
            link_blocks_cache: dict[object, list[object]] = {}
            for context in items:
                record = _materialize_record_payload(
                    context,
                    _record_from_context(context),
                    graph_cache=graph_cache,
                    node_blocks_cache=node_blocks_cache,
                    link_blocks_cache=link_blocks_cache,
                )
                if isinstance(record, list):
                    records.extend(record)
                else:
                    records.append(record)
            return LinkPredictionBatch.from_records(records)
        if kind == "temporal":
            records = []
            graph_cache: dict[object, Graph] = {}
            node_blocks_cache: dict[object, list[object]] = {}
            link_blocks_cache: dict[object, list[object]] = {}
            for context in items:
                record = _materialize_record_payload(
                    context,
                    _record_from_context(context),
                    graph_cache=graph_cache,
                    node_blocks_cache=node_blocks_cache,
                    link_blocks_cache=link_blocks_cache,
                )
                if isinstance(record, list):
                    records.extend(record)
                else:
                    records.append(record)
            return TemporalEventBatch.from_records(records)
        raise ValueError(f"unsupported context request kind: {kind}")

    if isinstance(first, TemporalEventRecord):
        return TemporalEventBatch.from_records(items)
    if isinstance(first, LinkPredictionRecord):
        return LinkPredictionBatch.from_records(items)
    if isinstance(first, SampleRecord) and first.subgraph_seed is not None and label_source is None:
        return NodeBatch.from_samples(items)
    if isinstance(first, SampleRecord):
        return GraphBatch.from_samples(items, label_key=label_key, label_source=label_source)
    if hasattr(first, "graph") and label_source is not None and label_key is not None:
        return GraphBatch.from_samples(items, label_key=label_key, label_source=label_source)
    return GraphBatch.from_graphs(items)
