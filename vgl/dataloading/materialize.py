from dataclasses import replace

import torch

from vgl.dataloading.executor import MaterializationContext
from vgl.dataloading.records import LinkPredictionRecord, SampleRecord, TemporalEventRecord
from vgl.graph.batch import (
    GraphBatch,
    LinkPredictionBatch,
    NodeBatch,
    TemporalEventBatch,
    _without_supervision_edges,
    _without_supervision_edges_for_type,
)
from vgl.graph.graph import Graph
from vgl.ops.block import to_block


def _align_tensor_slice(index: torch.Tensor, tensor_slice) -> torch.Tensor:
    index = torch.as_tensor(index, dtype=torch.long).view(-1)
    slice_index = tensor_slice.index.to(dtype=torch.long).view(-1)
    if index.numel() == 0:
        return tensor_slice.values.new_empty((0,) + tuple(tensor_slice.values.shape[1:]))
    if torch.equal(index, slice_index):
        return tensor_slice.values
    positions = {int(value): current for current, value in enumerate(slice_index.tolist())}
    order = []
    for value in index.tolist():
        value = int(value)
        if value not in positions:
            raise KeyError(f"fetched feature slice is missing id {value}")
        order.append(positions[value])
    order_index = torch.tensor(order, dtype=torch.long, device=tensor_slice.values.device)
    return tensor_slice.values[order_index]


def _subgraph(
    graph: Graph,
    node_ids: torch.Tensor,
    *,
    fetched_node_features: dict[str, dict[str, object]] | None = None,
    fetched_edge_features: dict[object, dict[str, object]] | None = None,
):
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
    for feature_name, tensor_slice in (fetched_node_features or {}).get("node", {}).items():
        node_data[feature_name] = _align_tensor_slice(node_ids, tensor_slice)

    edge_type = graph._default_edge_type()
    edge_store = graph.edges[edge_type]
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
    for feature_name, tensor_slice in (fetched_edge_features or {}).get(edge_type, {}).items():
        edge_data[feature_name] = _align_tensor_slice(edge_ids, tensor_slice)
    return Graph.homo(edge_index=subgraph_edge_index, edge_data=edge_data, **node_data), node_mapping


def _hetero_subgraph(
    graph: Graph,
    node_ids_by_type: dict[str, torch.Tensor],
    *,
    fetched_node_features: dict[str, dict[str, object]] | None = None,
    fetched_edge_features: dict[object, dict[str, object]] | None = None,
):
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
        for feature_name, tensor_slice in (fetched_node_features or {}).get(node_type, {}).items():
            node_data[feature_name] = _align_tensor_slice(node_ids, tensor_slice)
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
        for feature_name, tensor_slice in (fetched_edge_features or {}).get(edge_type, {}).items():
            edge_data[feature_name] = _align_tensor_slice(edge_ids, tensor_slice)
        edges[edge_type] = edge_data
    return Graph.hetero(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr), node_mappings


def _node_store_shape(store) -> tuple[int, torch.device]:
    for value in store.data.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.size(0)), value.device
    raise ValueError(f"cannot infer node shape for node type {store.type_name!r}")


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
            node_ids = torch.arange(graph.x.size(0), dtype=torch.long, device=graph.x.device)
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
                node_count, device = _node_store_shape(store)
                node_ids = torch.arange(node_count, dtype=torch.long, device=device)
                node_data["n_id"] = node_ids
            for feature_name, tensor_slice in (fetched_node_features or {}).get(node_type, {}).items():
                node_data[feature_name] = _align_tensor_slice(node_ids, tensor_slice)
            nodes[node_type] = node_data

        edges = {}
        for edge_type, store in graph.edges.items():
            edge_data = dict(store.data)
            edge_ids = edge_data.get("e_id")
            if edge_ids is None:
                edge_ids = torch.arange(store.edge_index.size(1), dtype=torch.long, device=store.edge_index.device)
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


def _build_homo_blocks(subgraph: Graph, node_mapping: torch.Tensor, node_hops: list[torch.Tensor]):
    subgraph_hops = []
    for hop_node_ids in node_hops:
        hop_node_ids = torch.as_tensor(hop_node_ids, dtype=torch.long, device=node_mapping.device).view(-1)
        subgraph_hops.append(node_mapping[hop_node_ids])
    return [to_block(subgraph, dst_nodes) for dst_nodes in subgraph_hops[-2::-1]]


def _build_homo_blocks_from_local_ids(subgraph: Graph, node_ids_local: torch.Tensor, node_hops: list[torch.Tensor]):
    node_ids_local = torch.as_tensor(node_ids_local, dtype=torch.long).view(-1)
    positions = {int(node_id): index for index, node_id in enumerate(node_ids_local.tolist())}
    subgraph_hops = []
    for hop_node_ids in node_hops:
        hop_node_ids = torch.as_tensor(hop_node_ids, dtype=torch.long).view(-1)
        subgraph_hops.append(
            torch.tensor(
                [positions[int(node_id)] for node_id in hop_node_ids.tolist()],
                dtype=torch.long,
                device=node_ids_local.device,
            )
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
        device = next(iter(subgraph.nodes[dst_type].data.values())).device
        dst_node_ids = torch.arange(node_count, dtype=torch.long, device=device)
    dst_node_ids = torch.as_tensor(dst_node_ids, dtype=torch.long).view(-1)
    positions = {int(node_id): index for index, node_id in enumerate(dst_node_ids.tolist())}
    subgraph_hops = []
    for hop_nodes in node_hops_by_type:
        hop_node_ids = torch.as_tensor(hop_nodes.get(dst_type, ()), dtype=torch.long).view(-1)
        subgraph_hops.append(
            torch.tensor(
                [positions[int(node_id)] for node_id in hop_node_ids.tolist()],
                dtype=torch.long,
                device=dst_node_ids.device,
            )
        )
    return [to_block(subgraph, dst_nodes, edge_type=edge_type) for dst_nodes in subgraph_hops[-2::-1]]


def _link_message_passing_graph(graph: Graph, records: list[LinkPredictionRecord]) -> Graph:
    supervision_edges_by_type = {}
    reverse_supervision_edges = {}
    for record in records:
        if int(record.label) != 1:
            continue
        if not (bool(getattr(record, "exclude_seed_edge", False)) or bool(record.metadata.get("exclude_seed_edges", False))):
            continue
        edge_type = getattr(record, "edge_type", None) or record.metadata.get("edge_type")
        if edge_type is None:
            edge_type = graph._default_edge_type()
        edge_type = tuple(edge_type)
        supervision_edges_by_type.setdefault(edge_type, set()).add((int(record.src_index), int(record.dst_index)))
        reverse_edge_type = getattr(record, "reverse_edge_type", None)
        if reverse_edge_type is None:
            reverse_edge_type = record.metadata.get("reverse_edge_type")
        if reverse_edge_type is not None:
            reverse_edge_type = tuple(reverse_edge_type)
            reverse_supervision_edges.setdefault(reverse_edge_type, set()).add((int(record.dst_index), int(record.src_index)))
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


def _materialize_record_payload(context: MaterializationContext, payload):
    fetched_node_features = context.state.get("_materialized_node_features")
    fetched_edge_features = context.state.get("_materialized_edge_features")
    needs_homo_node_blocks = "node_hops" in context.state
    needs_hetero_node_blocks = "node_hops_by_type" in context.state and "node_block_edge_type" in context.state
    needs_node_blocks = needs_homo_node_blocks or needs_hetero_node_blocks
    needs_homo_link_blocks = "link_node_hops" in context.state and "link_node_ids_local" in context.state
    needs_hetero_link_blocks = "link_node_hops_by_type" in context.state and "link_block_edge_type" in context.state
    needs_link_blocks = needs_homo_link_blocks or needs_hetero_link_blocks
    if not fetched_node_features and not fetched_edge_features and not needs_node_blocks and not needs_link_blocks:
        return payload

    graph_cache = {}
    node_blocks_cache = {}
    link_blocks_cache = {}

    def _materialized_graph(graph):
        graph_id = id(graph)
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
        cache_key = (id(graph), context.state.get("node_block_edge_type"))
        blocks = node_blocks_cache.get(cache_key)
        if blocks is None:
            if needs_homo_node_blocks:
                node_ids = graph.nodes["node"].data.get("n_id")
                if node_ids is None:
                    node_ids = torch.arange(graph.x.size(0), dtype=torch.long, device=graph.x.device)
                blocks = _build_homo_blocks_from_local_ids(graph, node_ids, context.state["node_hops"])
            else:
                blocks = _build_hetero_blocks_from_local_ids(
                    graph,
                    context.state["node_hops_by_type"],
                    edge_type=context.state["node_block_edge_type"],
                )
            node_blocks_cache[cache_key] = blocks
        return blocks

    def _materialized_link_blocks(records: list[LinkPredictionRecord], graph: Graph):
        cache_key = (id(graph), context.state.get("link_block_edge_type"))
        blocks = link_blocks_cache.get(cache_key)
        if blocks is None:
            message_passing_graph = _link_message_passing_graph(graph, records)
            if needs_homo_link_blocks:
                blocks = _build_homo_blocks_from_local_ids(
                    message_passing_graph,
                    context.state["link_node_ids_local"],
                    context.state["link_node_hops"],
                )
            else:
                blocks = _build_hetero_blocks_from_local_ids(
                    message_passing_graph,
                    context.state["link_node_hops_by_type"],
                    edge_type=context.state["link_block_edge_type"],
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


def _node_context_to_sample(context: MaterializationContext) -> SampleRecord | list[SampleRecord]:
    if "sample" in context.state:
        return _materialize_record_payload(context, context.state["sample"])
    if context.graph is None:
        raise ValueError("node context requires graph for materialization")
    request = context.request
    metadata = dict(getattr(request, "metadata", {}))
    sample_id = context.metadata.get("sample_id", metadata.get("sample_id"))
    source_graph_id = context.metadata.get("source_graph_id", metadata.get("source_graph_id"))
    seeds = [int(value) for value in request.node_ids.reshape(-1).tolist()]
    if not seeds:
        raise ValueError("node context requires at least one seed for materialization")

    fetched_node_features = context.state.get("_materialized_node_features")
    fetched_edge_features = context.state.get("_materialized_edge_features")

    if "node_ids" in context.state:
        subgraph, node_mapping = _subgraph(
            context.graph,
            context.state["node_ids"],
            fetched_node_features=fetched_node_features,
            fetched_edge_features=fetched_edge_features,
        )
        subgraph_seeds = [int(node_mapping[seed].item()) for seed in seeds]
        node_type = "node"
    elif "node_ids_by_type" in context.state:
        node_type = request.node_type
        subgraph, node_mapping = _hetero_subgraph(
            context.graph,
            context.state["node_ids_by_type"],
            fetched_node_features=fetched_node_features,
            fetched_edge_features=fetched_edge_features,
        )
        subgraph_seeds = [int(node_mapping[node_type][seed].item()) for seed in seeds]
    else:
        raise ValueError("node context must contain expanded node ids for materialization")

    blocks = None
    if "node_hops" in context.state:
        if node_type != "node":
            raise ValueError("node block materialization currently supports homogeneous graphs only")
        blocks = _build_homo_blocks(subgraph, node_mapping, context.state["node_hops"])
    elif "node_hops_by_type" in context.state and "node_block_edge_type" in context.state:
        blocks = _build_hetero_blocks_from_local_ids(
            subgraph,
            context.state["node_hops_by_type"],
            edge_type=context.state["node_block_edge_type"],
        )

    samples = []
    for seed, subgraph_seed in zip(seeds, subgraph_seeds):
        sample_metadata = dict(metadata)
        sample_metadata["seed"] = seed
        if node_type != "node":
            sample_metadata.setdefault("node_type", node_type)
        samples.append(
            SampleRecord(
                graph=subgraph,
                metadata=sample_metadata,
                sample_id=sample_id,
                source_graph_id=source_graph_id,
                subgraph_seed=subgraph_seed,
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
    return SampleRecord(
        graph=graph,
        metadata=metadata,
        sample_id=context.metadata.get("sample_id", metadata.get("sample_id")),
        source_graph_id=context.metadata.get("source_graph_id", metadata.get("source_graph_id")),
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
            for context in items:
                sample = _node_context_to_sample(context)
                if isinstance(sample, list):
                    samples.extend(sample)
                else:
                    samples.append(sample)
            return NodeBatch.from_samples(samples)
        if kind == "graph":
            samples = [_graph_context_to_sample(context) for context in items]
            if label_source is not None and label_key is not None:
                return GraphBatch.from_samples(samples, label_source=label_source, label_key=label_key)
            return GraphBatch.from_graphs([sample.graph for sample in samples])
        if kind == "link":
            records = []
            for context in items:
                record = _materialize_record_payload(context, _record_from_context(context))
                if isinstance(record, list):
                    records.extend(record)
                else:
                    records.append(record)
            return LinkPredictionBatch.from_records(records)
        if kind == "temporal":
            records = []
            for context in items:
                record = _materialize_record_payload(context, _record_from_context(context))
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
    if hasattr(first, "graph") and label_source is not None and label_key is not None:
        return GraphBatch.from_samples(items, label_key=label_key, label_source=label_source)
    return GraphBatch.from_graphs(items)
