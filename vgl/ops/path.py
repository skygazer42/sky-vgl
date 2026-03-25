import torch

from vgl.graph.graph import Graph


LINE_GRAPH_EDGE_TYPE = ("node", "line", "node")


def _resolve_edge_type(graph: Graph, edge_type=None) -> tuple[str, str, str]:
    if edge_type is None:
        return graph._default_edge_type()
    return tuple(edge_type)


def _edge_ids(store) -> torch.Tensor:
    edge_count = int(store.edge_index.size(1))
    value = store.data.get("e_id")
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
        return torch.as_tensor(value, dtype=torch.long, device=store.edge_index.device)
    return torch.arange(edge_count, dtype=torch.long, device=store.edge_index.device)


def _normalize_seed_nodes(seeds, *, count: int, device: torch.device) -> torch.Tensor:
    seed_tensor = torch.as_tensor(seeds, dtype=torch.long, device=device).reshape(-1)
    if seed_tensor.numel() > 0 and ((seed_tensor < 0).any() or (seed_tensor >= count).any()):
        raise ValueError("seed nodes must fall within the start node type range")
    return seed_tensor


def _successor_map(edge_index: torch.Tensor) -> dict[int, torch.Tensor]:
    successors: dict[int, list[int]] = {}
    for src, dst in edge_index.t().tolist():
        successors.setdefault(int(src), []).append(int(dst))
    return {
        src: torch.tensor(dst_list, dtype=torch.long, device=edge_index.device)
        for src, dst_list in successors.items()
    }


def _sample_successors(current: torch.Tensor, successors: dict[int, torch.Tensor]) -> torch.Tensor:
    next_nodes = torch.full_like(current, -1)
    for index, node in enumerate(current.tolist()):
        if node < 0:
            continue
        neighbors = successors.get(int(node))
        if neighbors is None or neighbors.numel() == 0:
            continue
        choice = int(torch.randint(neighbors.numel(), (1,), device=neighbors.device).item())
        next_nodes[index] = neighbors[choice]
    return next_nodes


def line_graph(graph: Graph, *, edge_type=None, backtracking: bool = True, copy_edata: bool = True) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if src_type != dst_type:
        raise ValueError("line_graph requires matching source and destination node types")

    store = graph.edges[edge_type]
    edge_index = store.edge_index
    edge_pairs = [(int(src), int(dst)) for src, dst in edge_index.t().tolist()]

    line_edges = []
    for source_edge_id, (src, dst) in enumerate(edge_pairs):
        for target_edge_id, (next_src, next_dst) in enumerate(edge_pairs):
            if dst != next_src:
                continue
            if not backtracking and src == next_dst:
                continue
            line_edges.append((source_edge_id, target_edge_id))

    if line_edges:
        line_edge_index = torch.tensor(line_edges, dtype=torch.long, device=edge_index.device).t().contiguous()
    else:
        line_edge_index = torch.empty((2, 0), dtype=torch.long, device=edge_index.device)

    node_data = {"n_id": _edge_ids(store)}
    if copy_edata:
        edge_count = int(store.edge_index.size(1))
        for key, value in store.data.items():
            if key in {"edge_index", "e_id"}:
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                node_data[key] = value

    return Graph.hetero(
        nodes={"node": node_data},
        edges={LINE_GRAPH_EDGE_TYPE: {"edge_index": line_edge_index}},
    )


def random_walk(graph: Graph, seeds, *, length: int, edge_type=None) -> torch.Tensor:
    if length < 0:
        raise ValueError("length must be non-negative")

    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if length > 1 and src_type != dst_type:
        raise ValueError("random_walk requires a relation that composes with itself for multi-step walks")

    store = graph.edges[edge_type]
    device = store.edge_index.device
    seed_tensor = _normalize_seed_nodes(seeds, count=graph._node_count(src_type), device=device)
    traces = torch.full((seed_tensor.numel(), length + 1), -1, dtype=torch.long, device=device)
    if seed_tensor.numel() == 0:
        return traces

    traces[:, 0] = seed_tensor
    if length == 0:
        return traces

    successors = _successor_map(store.edge_index)
    current = seed_tensor
    for step in range(length):
        current = _sample_successors(current, successors)
        traces[:, step + 1] = current
    return traces


def _normalize_metapath(graph: Graph, metapath) -> tuple[tuple[str, str, str], ...]:
    normalized = tuple(tuple(edge_type) for edge_type in metapath)
    if not normalized:
        raise ValueError("metapath must contain at least one edge type")
    for edge_type in normalized:
        if edge_type not in graph.edges:
            raise KeyError(edge_type)
    for previous, current in zip(normalized, normalized[1:]):
        if previous[2] != current[0]:
            raise ValueError("metapath edge types must compose")
    return normalized


def _metapath_pairs(graph: Graph, metapath: tuple[tuple[str, str, str], ...]) -> list[tuple[int, int]]:
    current_pairs = [
        (int(src), int(dst))
        for src, dst in graph.edges[metapath[0]].edge_index.t().tolist()
    ]
    for edge_type in metapath[1:]:
        next_edges = {}
        for src, dst in graph.edges[edge_type].edge_index.t().tolist():
            next_edges.setdefault(int(src), []).append(int(dst))
        expanded = []
        for start, middle in current_pairs:
            for dst in next_edges.get(int(middle), ()): 
                expanded.append((int(start), int(dst)))
        current_pairs = expanded

    deduplicated = []
    seen = set()
    for pair in current_pairs:
        if pair in seen:
            continue
        seen.add(pair)
        deduplicated.append(pair)
    return deduplicated


def metapath_reachable_graph(graph: Graph, metapath, *, relation_name: str | None = None) -> Graph:
    normalized = _normalize_metapath(graph, metapath)
    start_type = normalized[0][0]
    end_type = normalized[-1][2]
    relation_name = relation_name or "__".join(relation for _, relation, _ in normalized)
    derived_edge_type = (start_type, relation_name, end_type)

    pairs = _metapath_pairs(graph, normalized)
    device = graph.edges[normalized[0]].edge_index.device
    if pairs:
        edge_index = torch.tensor(pairs, dtype=torch.long, device=device).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=device)

    nodes = {start_type: dict(graph.nodes[start_type].data)}
    if end_type != start_type:
        nodes[end_type] = dict(graph.nodes[end_type].data)
    return Graph.hetero(nodes=nodes, edges={derived_edge_type: {"edge_index": edge_index}})


def metapath_random_walk(graph: Graph, seeds, metapath) -> torch.Tensor:
    normalized = _normalize_metapath(graph, metapath)
    start_type = normalized[0][0]
    device = graph.edges[normalized[0]].edge_index.device
    seed_tensor = _normalize_seed_nodes(seeds, count=graph._node_count(start_type), device=device)
    traces = torch.full((seed_tensor.numel(), len(normalized) + 1), -1, dtype=torch.long, device=device)
    if seed_tensor.numel() == 0:
        return traces

    traces[:, 0] = seed_tensor
    current = seed_tensor
    for step, edge_type in enumerate(normalized, start=1):
        current = _sample_successors(current, _successor_map(graph.edges[edge_type].edge_index))
        traces[:, step] = current
    return traces
