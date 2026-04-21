import torch

from vgl._optional import import_optional
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


def _numeric_public_node_ids(node_records) -> torch.Tensor | None:
    values = []
    for node_id, _ in node_records:
        if isinstance(node_id, torch.Tensor):
            if node_id.numel() != 1:
                return None
            scalar = node_id.detach().cpu().reshape(()).item()
        else:
            scalar = node_id
        try:
            normalized = int(scalar)
        except Exception:
            return None
        if isinstance(scalar, float) and float(normalized) != float(scalar):
            return None
        values.append(normalized)
    return torch.tensor(values, dtype=torch.long)


def _numeric_public_edge_ids(edge_records) -> torch.Tensor | None:
    values = []
    for _, _, edge_key, _ in edge_records:
        if isinstance(edge_key, torch.Tensor):
            if edge_key.numel() != 1:
                return None
            scalar = edge_key.detach().cpu().reshape(()).item()
        else:
            scalar = edge_key
        try:
            normalized = int(scalar)
        except Exception:
            return None
        if isinstance(scalar, float) and float(normalized) != float(scalar):
            return None
        values.append(normalized)
    if len(set(values)) != len(values):
        return None
    return torch.tensor(values, dtype=torch.long)


def from_networkx(nx_graph):
    if not nx_graph.is_directed():
        raise ValueError("from_networkx requires a directed NetworkX graph")

    node_records = list(nx_graph.nodes(data=True))
    node_index = {node_id: position for position, (node_id, _) in enumerate(node_records)}
    node_data = _collect_feature_table(node_records, entity_kind="node")
    if "n_id" not in node_data:
        public_node_ids = _numeric_public_node_ids(node_records)
        if public_node_ids is not None:
            node_data["n_id"] = public_node_ids
        elif not node_data:
            node_data["n_id"] = torch.arange(len(node_records), dtype=torch.long)

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
    if nx_graph.is_multigraph() and "e_id" not in edge_data:
        public_edge_ids = _numeric_public_edge_ids(edge_records)
        if public_edge_ids is not None:
            edge_data["e_id"] = public_edge_ids
    return Graph.homo(edge_index=edge_index, edge_data=edge_data or None, **node_data)


def to_networkx(graph: Graph):
    nx = import_optional(
        "networkx",
        extra_name="networkx",
        feature_name="NetworkX interoperability",
    )

    _ensure_homo_graph(graph)
    nx_graph = nx.MultiDiGraph()
    node_store = graph.nodes["node"].data
    edge_store = graph.edges[("node", "to", "node")].data
    num_nodes = graph._node_count("node")
    edge_index = graph.edge_index
    edge_count = int(edge_index.size(1))
    public_ids = node_store.get("n_id")
    if isinstance(public_ids, torch.Tensor) and public_ids.ndim == 1 and int(public_ids.size(0)) == num_nodes:
        node_labels = public_ids.detach().cpu().numpy().reshape(-1)
        for node_id in node_labels:
            normalized = int(node_id)
            if float(normalized) != float(node_id):
                raise ValueError("node feature 'n_id' must contain integer public node ids for NetworkX export")
        if len({int(node_id) for node_id in node_labels}) != len(node_labels):
            raise ValueError("node feature 'n_id' must contain unique public node ids for NetworkX export")
    else:
        node_labels = list(range(num_nodes))

    for node_id in range(num_nodes):
        node_label = int(node_labels[node_id])
        nx_graph.add_node(node_label)
        for key, value in node_store.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == num_nodes:
                nx_graph.nodes[node_label][key] = value[node_id].detach().clone()

    edge_rows = edge_index.t().detach().cpu().numpy()
    for edge_id, (src_index, dst_index) in enumerate(edge_rows):
        src = int(node_labels[int(src_index)])
        dst = int(node_labels[int(dst_index)])
        edge_attrs = {}
        for key, value in edge_store.items():
            if key == "edge_index":
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_attrs[key] = value[edge_id].detach().clone()
        nx_graph.add_edge(src, dst, **edge_attrs)
    return nx_graph
