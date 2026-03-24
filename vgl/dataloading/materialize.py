import torch

from vgl.dataloading.executor import MaterializationContext
from vgl.dataloading.records import LinkPredictionRecord, SampleRecord, TemporalEventRecord
from vgl.graph.batch import GraphBatch, LinkPredictionBatch, NodeBatch, TemporalEventBatch
from vgl.graph.graph import Graph


def _subgraph(graph: Graph, node_ids: torch.Tensor):
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


def _hetero_subgraph(graph: Graph, node_ids_by_type: dict[str, torch.Tensor]):
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


def _node_context_to_sample(context: MaterializationContext) -> SampleRecord:
    if "sample" in context.state:
        return context.state["sample"]
    if context.graph is None:
        raise ValueError("node context requires graph for materialization")
    request = context.request
    metadata = dict(getattr(request, "metadata", {}))
    sample_id = context.metadata.get("sample_id", metadata.get("sample_id"))
    source_graph_id = context.metadata.get("source_graph_id", metadata.get("source_graph_id"))
    if request.node_ids.numel() != 1:
        raise ValueError("node context materialization currently supports one seed per context")
    seed = int(request.node_ids.reshape(-1)[0].item())

    if "node_ids" in context.state:
        subgraph, node_mapping = _subgraph(context.graph, context.state["node_ids"])
        subgraph_seed = int(node_mapping[seed].item())
    elif "node_ids_by_type" in context.state:
        node_type = request.node_type
        subgraph, node_mapping = _hetero_subgraph(context.graph, context.state["node_ids_by_type"])
        subgraph_seed = int(node_mapping[node_type][seed].item())
    else:
        raise ValueError("node context must contain expanded node ids for materialization")

    return SampleRecord(
        graph=subgraph,
        metadata=metadata,
        sample_id=sample_id,
        source_graph_id=source_graph_id,
        subgraph_seed=subgraph_seed,
    )


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
        return _record_from_context(context)
    raise ValueError(f"unsupported context request kind: {kind}")


def materialize_batch(items, *, label_source=None, label_key=None):
    if not items:
        raise ValueError("materialize_batch requires at least one item")

    first = items[0]
    if isinstance(first, MaterializationContext):
        kind = getattr(first.request, "kind", None)
        if kind == "node":
            return NodeBatch.from_samples([_node_context_to_sample(context) for context in items])
        if kind == "graph":
            samples = [_graph_context_to_sample(context) for context in items]
            if label_source is not None and label_key is not None:
                return GraphBatch.from_samples(samples, label_source=label_source, label_key=label_key)
            return GraphBatch.from_graphs([sample.graph for sample in samples])
        if kind == "link":
            records = []
            for context in items:
                record = _record_from_context(context)
                if isinstance(record, list):
                    records.extend(record)
                else:
                    records.append(record)
            return LinkPredictionBatch.from_records(records)
        if kind == "temporal":
            records = []
            for context in items:
                record = _record_from_context(context)
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
