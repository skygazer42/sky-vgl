import weakref

import torch

_LOOKUP_CACHE_MAX_ENTRIES = 64
_endpoint_lookup_cache: dict[tuple[int, int], dict[str, object]] = {}


def _tensor_signature(tensor: torch.Tensor) -> tuple[object, ...]:
    return (
        tuple(int(dim) for dim in tensor.shape),
        str(tensor.dtype),
        str(tensor.device),
        int(getattr(tensor, "_version", 0)),
    )


def _endpoint_lookup(edge_index, *, endpoint: int) -> tuple[torch.Tensor, torch.Tensor]:
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    signature = _tensor_signature(edge_index)
    key = (id(edge_index), int(endpoint))
    cached = _endpoint_lookup_cache.get(key)
    if cached is not None and cached["tensor_ref"]() is edge_index and cached["signature"] == signature:
        _endpoint_lookup_cache[key] = _endpoint_lookup_cache.pop(key)
        return cached["lookup"]

    major = edge_index[endpoint]
    minor = edge_index[1 - endpoint]
    if major.numel() == 0:
        lookup = (major, minor)
    else:
        order = torch.argsort(major, stable=True)
        lookup = (major[order], minor[order])

    if len(_endpoint_lookup_cache) >= _LOOKUP_CACHE_MAX_ENTRIES:
        _endpoint_lookup_cache.pop(next(iter(_endpoint_lookup_cache)))
    _endpoint_lookup_cache[key] = {
        "tensor_ref": weakref.ref(edge_index),
        "signature": signature,
        "lookup": lookup,
    }
    return lookup


def _expand_interval_values(values: torch.Tensor, counts: torch.Tensor, *, step: int) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_empty(0)

    positive = counts > 0
    if not bool(positive.all()):
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


def _grouped_neighbors(edge_index, frontier, *, endpoint: int) -> torch.Tensor:
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    frontier = torch.as_tensor(frontier, dtype=torch.long, device=edge_index.device).view(-1)
    if frontier.numel() == 0 or edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    sorted_major, sorted_minor = _endpoint_lookup(edge_index, endpoint=endpoint)
    frontier = torch.unique(frontier)
    starts = torch.searchsorted(sorted_major, frontier, right=False)
    ends = torch.searchsorted(sorted_major, frontier, right=True)
    counts = ends - starts
    keep = counts > 0
    if not bool(keep.any()):
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    starts = starts[keep]
    counts = counts[keep]
    positions = _expand_interval_values(starts, counts, step=1)
    return sorted_minor[positions]


def _exclude_members(candidates, visited) -> torch.Tensor:
    candidates = torch.as_tensor(candidates, dtype=torch.long).view(-1)
    if candidates.numel() == 0:
        return candidates
    candidates = torch.unique(candidates)

    visited = torch.as_tensor(visited, dtype=torch.long, device=candidates.device).view(-1)
    if visited.numel() == 0:
        return candidates
    visited = torch.unique(visited)

    positions = torch.searchsorted(visited, candidates)
    capped_positions = positions.clamp_max(visited.numel() - 1)
    found = (positions < visited.numel()) & (visited[capped_positions] == candidates)
    return candidates[~found]


def _directional_neighbor_candidates(edge_index, frontier, visited, *, endpoint: int) -> torch.Tensor:
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    frontier = torch.as_tensor(frontier, dtype=torch.long, device=edge_index.device).view(-1)
    visited = torch.as_tensor(visited, dtype=torch.long, device=edge_index.device).view(-1)
    if frontier.numel() == 0 or edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    return _exclude_members(_grouped_neighbors(edge_index, frontier, endpoint=endpoint), visited)


def _sample_homo_next_frontier(edge_index, frontier, visited, fanout, *, generator):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    frontier = torch.as_tensor(frontier, dtype=torch.long, device=edge_index.device).view(-1)
    visited = torch.as_tensor(visited, dtype=torch.long, device=edge_index.device).view(-1)
    if frontier.numel() == 0 or edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    neighbor_chunks = []
    src_neighbors = _grouped_neighbors(edge_index, frontier, endpoint=0)
    if src_neighbors.numel() > 0:
        neighbor_chunks.append(src_neighbors)
    dst_neighbors = _grouped_neighbors(edge_index, frontier, endpoint=1)
    if dst_neighbors.numel() > 0:
        neighbor_chunks.append(dst_neighbors)
    if not neighbor_chunks:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)

    candidate_nodes = _exclude_members(torch.cat(neighbor_chunks), visited)
    if fanout != -1 and candidate_nodes.numel() > fanout:
        permutation = torch.randperm(
            candidate_nodes.numel(),
            generator=generator,
            device=candidate_nodes.device,
        )[:fanout]
        candidate_nodes = candidate_nodes[permutation]
    return candidate_nodes


def _sample_homo_node_ids(edge_index, seed_nodes, num_neighbors, *, generator, return_hops: bool = False):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    visited = torch.unique(
        torch.as_tensor(seed_nodes, dtype=torch.long, device=edge_index.device).view(-1)
    )
    frontier = visited
    hop_nodes = [visited.clone()] if return_hops else None
    for fanout in num_neighbors:
        frontier = _sample_homo_next_frontier(
            edge_index,
            frontier,
            visited,
            fanout,
            generator=generator,
        )
        if frontier.numel() > 0:
            visited = torch.unique(torch.cat((visited, frontier)))
        if hop_nodes is not None:
            hop_nodes.append(visited.clone())
        elif frontier.numel() == 0:
            break
    if hop_nodes is not None:
        return visited, hop_nodes
    return visited


def _sample_fanout(candidates, fanout, *, generator):
    candidates = torch.as_tensor(candidates, dtype=torch.long).view(-1)
    if fanout != -1 and candidates.numel() > fanout:
        permutation = torch.randperm(
            candidates.numel(),
            generator=generator,
            device=candidates.device,
        )[:fanout]
        candidates = candidates[permutation]
    return candidates


def _relation_neighbor_candidates(edge_index, frontier, visited, *, src_type, dst_type):
    edge_index = torch.as_tensor(edge_index, dtype=torch.long)
    unique_types = tuple(dict.fromkeys((src_type, dst_type)))
    empty = {
        node_type: torch.empty(0, dtype=torch.long, device=edge_index.device)
        for node_type in unique_types
    }
    if edge_index.numel() == 0:
        return empty

    src_frontier = frontier.get(src_type)
    dst_frontier = frontier.get(dst_type)
    if src_type == dst_type:
        candidate_chunks = []
        if src_frontier is not None and src_frontier.numel() > 0:
            src_frontier = src_frontier.to(device=edge_index.device)
            src_candidates = _grouped_neighbors(edge_index, src_frontier, endpoint=0)
            if src_candidates.numel() > 0:
                candidate_chunks.append(src_candidates)
            dst_candidates = _grouped_neighbors(edge_index, src_frontier, endpoint=1)
            if dst_candidates.numel() > 0:
                candidate_chunks.append(dst_candidates)
        if not candidate_chunks:
            return empty
        candidates = torch.cat(candidate_chunks)
        visited_tensor = visited.get(src_type)
        if visited_tensor is None:
            visited_tensor = torch.empty(0, dtype=torch.long, device=edge_index.device)
        return {src_type: _exclude_members(candidates, visited_tensor)}

    if dst_frontier is not None and dst_frontier.numel() > 0:
        src_candidates = _grouped_neighbors(
            edge_index,
            dst_frontier.to(device=edge_index.device),
            endpoint=1,
        )
    else:
        src_candidates = torch.empty(0, dtype=torch.long, device=edge_index.device)
    if src_frontier is not None and src_frontier.numel() > 0:
        dst_candidates = _grouped_neighbors(
            edge_index,
            src_frontier.to(device=edge_index.device),
            endpoint=0,
        )
    else:
        dst_candidates = torch.empty(0, dtype=torch.long, device=edge_index.device)

    visited_src = visited.get(src_type)
    if visited_src is None:
        visited_src = torch.empty(0, dtype=torch.long, device=edge_index.device)
    visited_dst = visited.get(dst_type)
    if visited_dst is None:
        visited_dst = torch.empty(0, dtype=torch.long, device=edge_index.device)
    return {
        src_type: _exclude_members(src_candidates, visited_src),
        dst_type: _exclude_members(dst_candidates, visited_dst),
    }
