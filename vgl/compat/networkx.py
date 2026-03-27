import torch

from vgl.graph import Graph


def _ensure_homo_graph(graph: Graph) -> None:
    if set(graph.nodes) != {"node"} or set(graph.edges) != {("node", "to", "node")}:
        raise ValueError("to_networkx currently supports homogeneous graphs only")


def _coerce_tensor(value, *, context: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    try:
        return torch.as_tensor(value)
    except Exception as exc:  # pragma: no cover - defensive path
        raise ValueError(f"{context} must be tensor-like for NetworkX interoperability") from exc


def _stack_feature_rows(values, *, context: str) -> torch.Tensor:
    rows = [_coerce_tensor(value, context=context) for value in values]
    if not rows:
        raise ValueError(f"{context} must include at least one value")
    try:
        return torch.stack(rows)
    except RuntimeError as exc:
        raise ValueError(f"{context} must have shape-compatible values") from exc


def _collect_feature_table(records, *, entity_kind: str) -> dict[str, torch.Tensor]:
    feature_keys = tuple(dict.fromkeys(key for _, attrs in records for key in attrs))
    table = {}
    for key in feature_keys:
        values = []
        for _, attrs in records:
            if key not in attrs:
                raise ValueError(f"{entity_kind} attribute {key!r} must exist on every {entity_kind}")
            values.append(attrs[key])
        table[key] = _stack_feature_rows(values, context=f"{entity_kind} attribute {key!r}")
    return table


def from_networkx(nx_graph):
    if not nx_graph.is_directed():
        raise ValueError("from_networkx requires a directed NetworkX graph")

    node_records = list(nx_graph.nodes(data=True))
    node_index = {node_id: position for position, (node_id, _) in enumerate(node_records)}
    node_data = _collect_feature_table(node_records, entity_kind="node")

    if nx_graph.is_multigraph():
        edge_records = list(nx_graph.edges(keys=True, data=True))
        edge_pairs = [(src, dst) for src, dst, _, _ in edge_records]
        edge_attr_rows = [attrs for _, _, _, attrs in edge_records]
    else:
        edge_records = list(nx_graph.edges(data=True))
        edge_pairs = [(src, dst) for src, dst, _ in edge_records]
        edge_attr_rows = [attrs for _, _, attrs in edge_records]

    if edge_pairs:
        edge_index = torch.tensor(
            [[node_index[src] for src, _ in edge_pairs], [node_index[dst] for _, dst in edge_pairs]],
            dtype=torch.long,
        )
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_data = _collect_feature_table(list(enumerate(edge_attr_rows)), entity_kind="edge")
    return Graph.homo(edge_index=edge_index, edge_data=edge_data or None, **node_data)


def to_networkx(graph: Graph):
    import networkx as nx  # type: ignore[import-not-found]

    _ensure_homo_graph(graph)
    nx_graph = nx.MultiDiGraph()
    node_store = graph.nodes["node"].data
    edge_store = graph.edges[("node", "to", "node")].data
    num_nodes = graph._node_count("node")
    edge_index = graph.edge_index
    edge_count = int(edge_index.size(1))

    for node_id in range(num_nodes):
        nx_graph.add_node(node_id)
        for key, value in node_store.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == num_nodes:
                nx_graph.nodes[node_id][key] = value[node_id].detach().clone()

    for edge_id in range(edge_count):
        src = int(edge_index[0, edge_id])
        dst = int(edge_index[1, edge_id])
        edge_attrs = {}
        for key, value in edge_store.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_attrs[key] = value[edge_id].detach().clone()
        nx_graph.add_edge(src, dst, **edge_attrs)
    return nx_graph
