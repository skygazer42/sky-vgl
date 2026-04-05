import torch

from vgl._neighbor_sampling import (
    _directional_neighbor_candidates,
    _relation_neighbor_candidates,
    _sample_fanout,
    _sample_homo_node_ids,
)
from vgl.graph.graph import Graph
from vgl.ops.subgraph import _ordered_unique, node_subgraph


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


def _normalize_hetero_khop_seeds(seeds, *, edge_type, device):
    if not isinstance(seeds, dict):
        raise ValueError("heterogeneous khop_nodes requires seeds keyed by node type")

    src_type, _, dst_type = edge_type
    allowed_types = {src_type, dst_type}
    unexpected = sorted(set(seeds) - allowed_types)
    if unexpected:
        joined = ", ".join(unexpected)
        raise ValueError(f"heterogeneous khop_nodes received seed types outside the selected relation: {joined}")

    normalized = {}
    for node_type in dict.fromkeys((src_type, dst_type)):
        values = seeds.get(node_type, torch.empty(0, dtype=torch.long, device=device))
        normalized[node_type] = _ordered_unique(values).to(device=device)
    return normalized


def _node_store_device(graph: Graph, node_type: str) -> torch.device:
    for value in graph.nodes[node_type].data.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def _hetero_directional_khop_nodes(graph: Graph, seeds, *, num_hops: int, direction: str, edge_type):
    src_type, _, dst_type = edge_type
    device = graph.edges[edge_type].edge_index.device
    normalized_seeds = _normalize_hetero_khop_seeds(seeds, edge_type=edge_type, device=device)
    edge_index = graph.edges[edge_type].edge_index

    visited = {
        node_type: torch.unique(node_ids.to(device=device))
        for node_type, node_ids in normalized_seeds.items()
    }
    frontier = {
        node_type: node_ids.clone()
        for node_type, node_ids in visited.items()
        if node_ids.numel() > 0
    }

    for _ in range(num_hops):
        if direction == "out":
            current = frontier.get(src_type)
            if current is None or current.numel() == 0 or edge_index.numel() == 0:
                break
            next_dst = _directional_neighbor_candidates(
                edge_index,
                current.to(device=device),
                visited[dst_type],
                endpoint=0,
            )
            if next_dst.numel() == 0:
                break
            frontier = {dst_type: next_dst}
            visited[dst_type] = torch.unique(torch.cat((visited[dst_type], next_dst)))
        else:
            current = frontier.get(dst_type)
            if current is None or current.numel() == 0 or edge_index.numel() == 0:
                break
            next_src = _directional_neighbor_candidates(
                edge_index,
                current.to(device=device),
                visited[src_type],
                endpoint=1,
            )
            if next_src.numel() == 0:
                break
            frontier = {src_type: next_src}
            visited[src_type] = torch.unique(torch.cat((visited[src_type], next_src)))

    return {
        node_type: visited[node_type].clone()
        for node_type in dict.fromkeys((src_type, dst_type))
    }


def expand_neighbors(
    graph: Graph,
    seeds,
    *,
    num_neighbors,
    node_type: str | None = None,
    generator=None,
    return_hops: bool = False,
):
    fanouts = _normalize_fanouts(num_neighbors)
    seed_tensor = torch.as_tensor(seeds, dtype=torch.long).view(-1)

    if set(graph.nodes) == {"node"} and len(graph.edges) == 1:
        return _sample_homo_node_ids(
            graph.edge_index,
            seed_tensor,
            fanouts,
            generator=generator,
            return_hops=return_hops,
        )

    if node_type is None:
        raise ValueError("node_type is required for heterogeneous neighbor expansion")
    if node_type not in graph.nodes:
        raise ValueError("node_type must exist in the source graph")

    def _snapshot(visited_by_type):
        return {
            current_type: node_ids.clone()
            for current_type, node_ids in visited_by_type.items()
        }

    visited = {
        current_type: torch.empty(
            0,
            dtype=torch.long,
            device=_node_store_device(graph, current_type),
        )
        for current_type in graph.schema.node_types
    }
    visited[node_type] = torch.unique(seed_tensor.to(device=visited[node_type].device))
    frontier = {node_type: visited[node_type].clone()}
    hop_nodes = [_snapshot(visited)] if return_hops else None
    for fanout in fanouts:
        candidates: dict[str, list[torch.Tensor]] = {
            current_type: [] for current_type in graph.schema.node_types
        }
        for edge_type, store in graph.edges.items():
            src_type, _, dst_type = edge_type
            src_frontier = frontier.get(src_type)
            dst_frontier = frontier.get(dst_type)
            if (src_frontier is None or src_frontier.numel() == 0) and (
                dst_frontier is None or dst_frontier.numel() == 0
            ):
                continue
            relation_candidates = _relation_neighbor_candidates(
                store.edge_index,
                frontier,
                visited,
                src_type=src_type,
                dst_type=dst_type,
            )
            for current_type, node_ids in relation_candidates.items():
                if node_ids.numel() > 0:
                    candidates[current_type].append(node_ids)

        next_frontier = {}
        for current_type, node_ids_list in candidates.items():
            if not node_ids_list:
                continue
            candidate_tensor = torch.unique(torch.cat(node_ids_list))
            candidate_tensor = _sample_fanout(candidate_tensor, fanout, generator=generator)
            if candidate_tensor.numel() > 0:
                next_frontier[current_type] = candidate_tensor
                visited[current_type] = torch.unique(torch.cat((visited[current_type], candidate_tensor)))
        if hop_nodes is not None:
            hop_nodes.append(_snapshot(visited))
        elif not next_frontier:
            break
        frontier = next_frontier
    expanded = {
        current_type: node_ids.clone()
        for current_type, node_ids in visited.items()
    }
    if hop_nodes is not None:
        return expanded, hop_nodes
    return expanded


def khop_nodes(graph: Graph, seeds, *, num_hops: int, direction: str = "out", edge_type=None) -> torch.Tensor | dict[str, torch.Tensor]:
    if num_hops < 0:
        raise ValueError("num_hops must be >= 0")
    if direction not in {"out", "in"}:
        raise ValueError("direction must be 'out' or 'in'")
    edge_type = _resolve_edge_type(graph, edge_type)
    src_type, _, dst_type = edge_type
    edge_index = graph.edges[edge_type].edge_index

    if src_type != dst_type:
        return _hetero_directional_khop_nodes(
            graph,
            seeds,
            num_hops=num_hops,
            direction=direction,
            edge_type=edge_type,
        )

    frontier = torch.unique(torch.as_tensor(seeds, dtype=torch.long, device=edge_index.device).view(-1))
    visited = frontier.clone()
    for _ in range(num_hops):
        if frontier.numel() == 0 or edge_index.numel() == 0:
            break
        if direction == "out":
            next_nodes = _directional_neighbor_candidates(
                edge_index,
                frontier,
                visited,
                endpoint=0,
            )
        else:
            next_nodes = _directional_neighbor_candidates(
                edge_index,
                frontier,
                visited,
                endpoint=1,
            )
        if next_nodes.numel() == 0:
            break
        frontier = next_nodes
        visited = torch.unique(torch.cat((visited, next_nodes)))
    return visited


def khop_subgraph(graph: Graph, seeds, *, num_hops: int, direction: str = "out", edge_type=None) -> Graph:
    nodes = khop_nodes(graph, seeds, num_hops=num_hops, direction=direction, edge_type=edge_type)
    return node_subgraph(graph, nodes, edge_type=edge_type)
