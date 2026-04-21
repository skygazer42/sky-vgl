from __future__ import annotations

import torch

from vgl.ops.subgraph import node_subgraph
from vgl.transforms.base import BaseTransform
from vgl.transforms._utils import clone_graph, is_edge_aligned, is_homo_graph


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _edge_pair_keys(edge_index: torch.Tensor, *, dst_count: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=edge_index.device)
    return edge_index[0].to(dtype=torch.long) * int(dst_count) + edge_index[1].to(dtype=torch.long)


def _homo_num_nodes(graph) -> int:
    return int(graph._node_count("node"))


def _fresh_edge_ids(existing: torch.Tensor, *, count: int, device: torch.device) -> torch.Tensor:
    if count == 0:
        return torch.empty((0,), dtype=existing.dtype if existing.numel() > 0 else torch.long, device=device)
    if existing.numel() == 0:
        start = 0
        dtype = torch.long
    else:
        start = _as_python_int(existing.to(dtype=torch.long).max()) + 1
        dtype = existing.dtype
    return torch.arange(start, start + count, dtype=dtype, device=device)


def _sorted_unique_tensor(values) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    if values.numel() == 0:
        return values
    sorted_values = torch.sort(values, stable=True).values
    keep = torch.ones(sorted_values.numel(), dtype=torch.bool, device=sorted_values.device)
    if sorted_values.numel() > 1:
        keep[1:] = sorted_values[1:] != sorted_values[:-1]
    return sorted_values[keep]


def _sorted_unique_with_inverse(values) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    if values.numel() == 0:
        return values, values
    order = torch.argsort(values, stable=True)
    sorted_values = values.index_select(0, order)
    keep = torch.ones(sorted_values.numel(), dtype=torch.bool, device=sorted_values.device)
    if sorted_values.numel() > 1:
        keep[1:] = sorted_values[1:] != sorted_values[:-1]
    unique_values = sorted_values[keep]
    group_ids = torch.cumsum(keep.to(dtype=torch.long), dim=0) - 1
    inverse = torch.empty_like(group_ids)
    inverse[order] = group_ids
    return unique_values, inverse


def _membership_mask(values: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    allowed = torch.as_tensor(allowed, dtype=torch.long, device=values.device).view(-1)
    if values.numel() == 0 or allowed.numel() == 0:
        return torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    sorted_allowed = _sorted_unique_tensor(allowed)
    positions = torch.searchsorted(sorted_allowed, values, right=False)
    capped_positions = positions.clamp_max(sorted_allowed.numel() - 1)
    return (positions < sorted_allowed.numel()) & (sorted_allowed[capped_positions] == values)


def _stable_unique_positions(keys: torch.Tensor) -> torch.Tensor:
    if keys.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=keys.device)
    unique_keys, inverse = _sorted_unique_with_inverse(keys)
    positions = torch.arange(keys.numel(), dtype=torch.long, device=keys.device)
    first_positions = torch.full((unique_keys.numel(),), keys.numel(), dtype=torch.long, device=keys.device)
    first_positions.scatter_reduce_(0, inverse, positions, reduce="amin", include_self=True)
    return torch.sort(first_positions, stable=True).values


class ToUndirected(BaseTransform):
    def __call__(self, graph):
        if not is_homo_graph(graph):
            raise ValueError("ToUndirected currently supports homogeneous graphs only")
        edge_type = graph._default_edge_type()
        store = graph.edges[edge_type]
        edge_index = store.edge_index
        num_nodes = _homo_num_nodes(graph)
        reverse_index = edge_index[[1, 0]].contiguous()
        missing_mask = ~_membership_mask(
            _edge_pair_keys(reverse_index, dst_count=num_nodes),
            _edge_pair_keys(edge_index, dst_count=num_nodes),
        )
        if not bool(missing_mask.any()):
            return graph

        candidate_indices = torch.nonzero(missing_mask, as_tuple=False).view(-1)
        mirror_indices = candidate_indices[
            _stable_unique_positions(_edge_pair_keys(reverse_index[:, missing_mask], dst_count=num_nodes))
        ]
        if mirror_indices.numel() == 0:
            return graph

        added_edge_index = reverse_index[:, mirror_indices]
        new_edge_index = torch.cat([edge_index, added_edge_index], dim=1)
        edge_count = int(edge_index.size(1))
        edges = {edge_type: {"edge_index": new_edge_index}}
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if is_edge_aligned(value, edge_count):
                if key == "e_id":
                    appended = _fresh_edge_ids(value, count=int(mirror_indices.numel()), device=value.device)
                    edges[edge_type][key] = torch.cat([value, appended], dim=0)
                else:
                    edges[edge_type][key] = torch.cat([value, value[mirror_indices]], dim=0)
            else:
                edges[edge_type][key] = value
        return clone_graph(graph, edges=edges)


class AddSelfLoops(BaseTransform):
    def __init__(self, *, fill_value: float = 1.0):
        self.fill_value = fill_value

    def __call__(self, graph):
        if not is_homo_graph(graph):
            raise ValueError("AddSelfLoops currently supports homogeneous graphs only")
        edge_type = graph._default_edge_type()
        store = graph.edges[edge_type]
        edge_index = store.edge_index
        num_nodes = _homo_num_nodes(graph)
        loop_mask = edge_index[0] == edge_index[1]
        has_loop = torch.zeros(num_nodes, dtype=torch.bool, device=edge_index.device)
        if bool(loop_mask.any()):
            has_loop[edge_index[0, loop_mask].to(dtype=torch.long)] = True
        missing_nodes = torch.nonzero(~has_loop, as_tuple=False).view(-1).to(dtype=edge_index.dtype)
        if missing_nodes.numel() == 0:
            return graph

        added_edge_index = torch.stack([missing_nodes, missing_nodes], dim=0)
        new_edge_index = torch.cat([edge_index, added_edge_index], dim=1)
        edge_count = int(edge_index.size(1))
        add_count = int(missing_nodes.numel())
        edges = {edge_type: {"edge_index": new_edge_index}}
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if is_edge_aligned(value, edge_count):
                if key == "e_id":
                    fill_tensor = _fresh_edge_ids(value, count=add_count, device=value.device)
                else:
                    fill_shape = (add_count,) + tuple(value.shape[1:])
                    fill_tensor = value.new_full(fill_shape, self.fill_value)
                edges[edge_type][key] = torch.cat([value, fill_tensor], dim=0)
            else:
                edges[edge_type][key] = value
        return clone_graph(graph, edges=edges)


class RemoveSelfLoops(BaseTransform):
    def __call__(self, graph):
        if not is_homo_graph(graph):
            raise ValueError("RemoveSelfLoops currently supports homogeneous graphs only")
        edge_type = graph._default_edge_type()
        store = graph.edges[edge_type]
        keep_mask = store.edge_index[0] != store.edge_index[1]
        edge_count = int(store.edge_index.size(1))
        edges = {edge_type: {"edge_index": store.edge_index[:, keep_mask]}}
        for key, value in store.data.items():
            if key == "edge_index":
                continue
            if is_edge_aligned(value, edge_count):
                edges[edge_type][key] = value[keep_mask]
            else:
                edges[edge_type][key] = value
        return clone_graph(graph, edges=edges)


class LargestConnectedComponents(BaseTransform):
    def __init__(self, *, num_components: int = 1):
        self.num_components = _as_python_int(num_components)

    def __call__(self, graph):
        if not is_homo_graph(graph):
            raise ValueError("LargestConnectedComponents currently supports homogeneous graphs only")
        if self.num_components < 1:
            raise ValueError("num_components must be >= 1")

        num_nodes = _homo_num_nodes(graph)
        labels = torch.arange(num_nodes, dtype=torch.long, device=graph.edge_index.device)
        if graph.edge_index.numel() > 0:
            src_nodes = graph.edge_index[0].to(dtype=torch.long)
            dst_nodes = graph.edge_index[1].to(dtype=torch.long)
            undirected_src = torch.cat((src_nodes, dst_nodes), dim=0)
            undirected_dst = torch.cat((dst_nodes, src_nodes), dim=0)

            for _ in range(num_nodes):
                propagated = labels.clone()
                propagated.scatter_reduce_(
                    0,
                    undirected_dst,
                    labels.index_select(0, undirected_src),
                    reduce="amin",
                    include_self=True,
                )
                if torch.equal(propagated, labels):
                    break
                labels = propagated

        _, inverse = _sorted_unique_with_inverse(labels)
        counts = torch.bincount(inverse)
        selected_components = torch.zeros(counts.numel(), dtype=torch.bool, device=labels.device)
        selected_components[torch.argsort(counts, descending=True, stable=True)[: self.num_components]] = True
        kept_nodes = torch.nonzero(selected_components[inverse], as_tuple=False).view(-1)
        return node_subgraph(graph, kept_nodes)
