import torch

from vgl.graph.graph import Graph
from vgl.ops.subgraph import node_subgraph


def _resolve_edge_type(graph: Graph, edge_type=None) -> tuple[str, str, str]:
    if edge_type is None:
        return graph._default_edge_type()
    return tuple(edge_type)


def _normalize_fanouts(num_neighbors) -> list[int]:
    if isinstance(num_neighbors, int):
        num_neighbors = [num_neighbors]
    fanouts = [int(value) for value in num_neighbors]
    if not fanouts:
        raise ValueError("num_neighbors must contain at least one hop")
    if any(value < -1 or value == 0 for value in fanouts):
        raise ValueError("num_neighbors entries must be positive integers or -1")
    return fanouts


def _next_frontier_from_edge_index(edge_index, frontier, visited, fanout, *, generator=None):
    if not frontier or edge_index.numel() == 0:
        return set()
    frontier_tensor = torch.tensor(sorted(frontier), dtype=torch.long, device=edge_index.device)
    incident_mask = torch.isin(edge_index[0], frontier_tensor) | torch.isin(edge_index[1], frontier_tensor)
    if not incident_mask.any():
        return set()
    candidate_nodes = torch.unique(edge_index[:, incident_mask]).tolist()
    candidate_nodes = [int(node) for node in candidate_nodes if int(node) not in visited]
    if fanout != -1 and len(candidate_nodes) > fanout:
        permutation = torch.randperm(len(candidate_nodes), generator=generator)[:fanout].tolist()
        candidate_nodes = [candidate_nodes[index] for index in permutation]
    return set(candidate_nodes)


def _hetero_next_frontier(graph, frontier, visited, fanout, *, generator=None):
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
            permutation = torch.randperm(len(candidate_list), generator=generator)[:fanout].tolist()
            candidate_list = [candidate_list[index] for index in permutation]
        if candidate_list:
            next_frontier[node_type] = set(candidate_list)
    return next_frontier


def expand_neighbors(graph: Graph, seeds, *, num_neighbors, node_type: str | None = None, generator=None):
    fanouts = _normalize_fanouts(num_neighbors)
    seed_tensor = torch.as_tensor(seeds, dtype=torch.long).view(-1)

    if set(graph.nodes) == {"node"} and len(graph.edges) == 1:
        visited = {int(node) for node in seed_tensor.tolist()}
        frontier = set(visited)
        for fanout in fanouts:
            frontier = _next_frontier_from_edge_index(
                graph.edge_index,
                frontier,
                visited,
                fanout,
                generator=generator,
            )
            visited.update(frontier)
            if not frontier:
                break
        return torch.tensor(sorted(visited), dtype=torch.long, device=graph.edge_index.device)

    if node_type is None:
        raise ValueError("node_type is required for heterogeneous neighbor expansion")
    if node_type not in graph.nodes:
        raise ValueError("node_type must exist in the source graph")

    visited = {current_type: set() for current_type in graph.schema.node_types}
    frontier = {current_type: set() for current_type in graph.schema.node_types}
    visited[node_type].update(int(node) for node in seed_tensor.tolist())
    frontier[node_type].update(int(node) for node in seed_tensor.tolist())
    for fanout in fanouts:
        frontier = _hetero_next_frontier(graph, frontier, visited, fanout, generator=generator)
        for current_type, node_ids in frontier.items():
            visited[current_type].update(node_ids)
        if not any(frontier.values()):
            break
    return {
        current_type: torch.tensor(
            sorted(visited[current_type]),
            dtype=torch.long,
            device=next(iter(graph.nodes[current_type].data.values())).device,
        )
        for current_type in graph.schema.node_types
    }


def khop_nodes(graph: Graph, seeds, *, num_hops: int, direction: str = "out", edge_type=None) -> torch.Tensor:
    if num_hops < 0:
        raise ValueError("num_hops must be >= 0")
    if direction not in {"out", "in"}:
        raise ValueError("direction must be 'out' or 'in'")
    edge_type = _resolve_edge_type(graph, edge_type)
    edge_index = graph.edges[edge_type].edge_index
    frontier = torch.as_tensor(seeds, dtype=torch.long).view(-1)
    visited = set(int(node) for node in frontier.tolist())
    for _ in range(num_hops):
        frontier_set = set(int(node) for node in frontier.tolist())
        next_nodes = []
        for src, dst in edge_index.t().tolist():
            src = int(src)
            dst = int(dst)
            candidate = None
            if direction == "out" and src in frontier_set:
                candidate = dst
            if direction == "in" and dst in frontier_set:
                candidate = src
            if candidate is not None and candidate not in visited:
                visited.add(candidate)
                next_nodes.append(candidate)
        frontier = torch.tensor(next_nodes, dtype=torch.long, device=edge_index.device)
        if frontier.numel() == 0:
            break
    return torch.tensor(sorted(visited), dtype=torch.long, device=edge_index.device)


def khop_subgraph(graph: Graph, seeds, *, num_hops: int, direction: str = "out", edge_type=None) -> Graph:
    nodes = khop_nodes(graph, seeds, num_hops=num_hops, direction=direction, edge_type=edge_type)
    return node_subgraph(graph, nodes, edge_type=edge_type)
