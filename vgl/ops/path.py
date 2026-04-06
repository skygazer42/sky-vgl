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


def _successor_state(edge_index: torch.Tensor, *, num_src_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = edge_index.device
    counts = torch.zeros(num_src_nodes, dtype=torch.long, device=device)
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device), counts, torch.empty(0, dtype=torch.long, device=device)

    sort_perm = torch.argsort(edge_index[0], stable=True)
    sorted_src = edge_index[0].index_select(0, sort_perm)
    sorted_dst = edge_index[1].index_select(0, sort_perm)
    counts = torch.bincount(sorted_src, minlength=num_src_nodes)
    starts = torch.cumsum(counts, dim=0) - counts
    return starts, counts, sorted_dst


def _source_edge_state(edge_index: torch.Tensor, *, num_src_nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = edge_index.device
    counts = torch.zeros(num_src_nodes, dtype=torch.long, device=device)
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device), counts
    sort_perm = torch.argsort(edge_index[0], stable=True)
    sorted_src = edge_index[0].index_select(0, sort_perm)
    counts = torch.bincount(sorted_src, minlength=num_src_nodes)
    starts = torch.cumsum(counts, dim=0) - counts
    return sort_perm, starts, counts


def _expand_interval_positions(starts: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    starts = torch.as_tensor(starts, dtype=torch.long).view(-1)
    counts = torch.as_tensor(counts, dtype=torch.long, device=starts.device).view(-1)
    if starts.numel() == 0 or counts.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=starts.device)

    positive = counts > 0
    starts = starts[positive]
    counts = counts[positive]
    if starts.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=starts.device)

    offsets = torch.cumsum(counts, dim=0) - counts
    bases = starts - offsets
    deltas = torch.empty_like(bases)
    deltas[0] = bases[0]
    if bases.numel() > 1:
        deltas[1:] = bases[1:] - bases[:-1]
    expanded = torch.zeros(counts.sum(), dtype=torch.long, device=starts.device)
    expanded[offsets] = deltas
    expanded = torch.cumsum(expanded, dim=0)
    expanded += torch.arange(counts.sum(), dtype=torch.long, device=starts.device)
    return expanded


def _expand_interval_values(values: torch.Tensor, counts: torch.Tensor, *, step: int) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    counts = torch.as_tensor(counts, dtype=torch.long, device=values.device).view(-1)
    if values.numel() == 0 or counts.numel() == 0:
        return values.new_empty(0)

    positive = counts > 0
    values = values[positive]
    counts = counts[positive]
    if values.numel() == 0:
        return values.new_empty(0)

    offsets = torch.cumsum(counts, dim=0) - counts
    base = values - step * offsets
    deltas = torch.empty_like(base)
    deltas[0] = base[0]
    if base.numel() > 1:
        deltas[1:] = base[1:] - base[:-1]
    markers = torch.zeros(counts.sum(), dtype=values.dtype, device=values.device)
    markers[offsets] = deltas
    expanded = torch.cumsum(markers, dim=0)
    if step != 0:
        expanded = expanded + step * torch.arange(counts.sum(), dtype=values.dtype, device=values.device)
    return expanded


def _edge_pair_keys(edge_index: torch.Tensor, *, dst_count: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    return edge_index[0].to(dtype=torch.long) * int(dst_count) + edge_index[1].to(dtype=torch.long)


def _stable_unique_positions(keys: torch.Tensor) -> torch.Tensor:
    if keys.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=keys.device)
    sorted_order = torch.argsort(keys, stable=True)
    sorted_keys = keys.index_select(0, sorted_order)
    group_starts = torch.ones(sorted_keys.numel(), dtype=torch.bool, device=keys.device)
    group_starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    return torch.sort(sorted_order[group_starts], stable=True).values


def _sample_offsets(counts: torch.Tensor) -> torch.Tensor:
    offsets = torch.zeros_like(counts)
    if counts.numel() == 0:
        return offsets

    sorted_counts = torch.sort(counts, stable=True).values
    positive_counts = sorted_counts[sorted_counts > 0]
    if positive_counts.numel() == 0:
        return offsets

    group_starts = torch.ones(positive_counts.numel(), dtype=torch.bool, device=counts.device)
    group_starts[1:] = positive_counts[1:] != positive_counts[:-1]
    for degree_tensor in positive_counts[group_starts]:
        if bool(degree_tensor <= 0):
            continue
        mask = counts == degree_tensor
        offsets[mask] = torch.randint(
            high=degree_tensor,
            size=(mask.sum(),),
            device=counts.device,
        )
    return offsets


def _sample_successors(current: torch.Tensor, successor_state: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    starts, counts, sorted_dst = successor_state
    next_nodes = torch.full_like(current, -1)
    if current.numel() == 0 or counts.numel() == 0:
        return next_nodes

    valid_positions = torch.nonzero(current >= 0, as_tuple=False).view(-1)
    if valid_positions.numel() == 0:
        return next_nodes

    valid_nodes = current.index_select(0, valid_positions)
    neighbor_counts = counts.index_select(0, valid_nodes)
    has_neighbors = neighbor_counts > 0
    if not bool(has_neighbors.any()):
        return next_nodes

    target_positions = valid_positions[has_neighbors]
    target_nodes = valid_nodes[has_neighbors]
    target_counts = neighbor_counts[has_neighbors]
    target_offsets = starts.index_select(0, target_nodes) + _sample_offsets(target_counts)
    next_nodes[target_positions] = sorted_dst.index_select(0, target_offsets)
    return next_nodes


def line_graph(graph: Graph, *, edge_type=None, backtracking: bool = True, copy_edata: bool = True) -> Graph:
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    if src_type != dst_type:
        raise ValueError("line_graph requires matching source and destination node types")

    store = graph.edges[edge_type]
    edge_index = store.edge_index
    sorted_edge_ids, starts, counts = _source_edge_state(edge_index, num_src_nodes=graph._node_count(src_type))
    src_nodes = edge_index[0].to(dtype=torch.long)
    dst_nodes = edge_index[1].to(dtype=torch.long)
    source_edge_ids = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    target_counts = counts.index_select(0, dst_nodes)
    active_sources = target_counts > 0
    if bool(active_sources.any()):
        line_sources = _expand_interval_values(source_edge_ids[active_sources], target_counts[active_sources], step=0)
        target_positions = _expand_interval_positions(
            starts.index_select(0, dst_nodes[active_sources]),
            target_counts[active_sources],
        )
        line_targets = sorted_edge_ids.index_select(0, target_positions)
        if not backtracking:
            keep = dst_nodes.index_select(0, line_targets) != src_nodes.index_select(0, line_sources)
            line_sources = line_sources[keep]
            line_targets = line_targets[keep]
        line_edge_index = torch.stack((line_sources, line_targets), dim=0)
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

    successors = _successor_state(store.edge_index, num_src_nodes=graph._node_count(src_type))
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


def _metapath_pairs(graph: Graph, metapath: tuple[tuple[str, str, str], ...]) -> torch.Tensor:
    current_pairs = graph.edges[metapath[0]].edge_index.to(dtype=torch.long)
    for edge_type in metapath[1:]:
        if current_pairs.numel() == 0:
            return current_pairs
        next_edge_index = graph.edges[edge_type].edge_index.to(dtype=torch.long, device=current_pairs.device)
        starts, counts, sorted_dst = _successor_state(
            next_edge_index,
            num_src_nodes=graph._node_count(edge_type[0]),
        )
        middle_nodes = current_pairs[1]
        target_counts = counts.index_select(0, middle_nodes)
        active_pairs = target_counts > 0
        if not bool(active_pairs.any()):
            return current_pairs[:, :0]
        expanded_starts = current_pairs[0, active_pairs]
        expanded_positions = _expand_interval_positions(
            starts.index_select(0, middle_nodes[active_pairs]),
            target_counts[active_pairs],
        )
        current_pairs = torch.stack(
            (
                _expand_interval_values(expanded_starts, target_counts[active_pairs], step=0),
                sorted_dst.index_select(0, expanded_positions),
            ),
            dim=0,
        )

    representatives = _stable_unique_positions(
        _edge_pair_keys(current_pairs, dst_count=graph._node_count(metapath[-1][2]))
    )
    return current_pairs.index_select(1, representatives)


def metapath_reachable_graph(graph: Graph, metapath, *, relation_name: str | None = None) -> Graph:
    normalized = _normalize_metapath(graph, metapath)
    start_type = normalized[0][0]
    end_type = normalized[-1][2]
    relation_name = relation_name or "__".join(relation for _, relation, _ in normalized)
    derived_edge_type = (start_type, relation_name, end_type)

    edge_index = _metapath_pairs(graph, normalized)
    device = graph.edges[normalized[0]].edge_index.device
    edge_index = edge_index.to(device=device, dtype=torch.long)
    if edge_index.numel() == 0:
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
        current = _sample_successors(
            current,
            _successor_state(graph.edges[edge_type].edge_index, num_src_nodes=graph._node_count(edge_type[0])),
        )
        traces[:, step] = current
    return traces
