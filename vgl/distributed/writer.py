from pathlib import Path

import torch

from vgl.data.ondisk import serialize_graph
from vgl.distributed.partition import PartitionManifest, PartitionShard, save_partition_manifest
from vgl.graph.graph import Graph


DEFAULT_EDGE_TYPE = ("node", "to", "node")


def _tensor_int_tuple(values) -> tuple[int, ...]:
    tensor = torch.as_tensor(values, dtype=torch.long).view(-1)
    return tuple(int(value.item()) for value in tensor)


def _slice_node_data(graph: Graph, node_type: str, start: int, end: int, *, device) -> tuple[dict, torch.Tensor]:
    node_count = graph._node_count(node_type)
    node_data = {}
    for key, value in graph.nodes[node_type].data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == node_count:
            node_data[key] = value[start:end]
        else:
            node_data[key] = value
    node_ids = torch.arange(start, end, dtype=torch.long, device=device)
    node_data.setdefault("n_id", node_ids)
    return node_data, node_ids


def _slice_edge_data(edge_store, edge_mask: torch.Tensor, *, edge_index: torch.Tensor) -> dict:
    edge_count = int(edge_store.edge_index.size(1))
    edge_data = {"edge_index": edge_index}
    for key, value in edge_store.data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[edge_mask]
        else:
            edge_data[key] = value
    edge_data.setdefault(
        "e_id",
        torch.arange(edge_count, dtype=torch.long, device=edge_store.edge_index.device)[edge_mask],
    )
    return edge_data


def _build_partition_graph(graph: Graph, node_payloads: dict, edges: dict) -> Graph:
    if graph.schema.time_attr is not None:
        return Graph.temporal(nodes=node_payloads, edges=edges, time_attr=graph.schema.time_attr)
    if set(node_payloads) == {"node"} and set(edges) == {DEFAULT_EDGE_TYPE}:
        edge_payload = dict(edges[DEFAULT_EDGE_TYPE])
        edge_index = edge_payload.pop("edge_index")
        return Graph.homo(edge_index=edge_index, edge_data=edge_payload, **node_payloads["node"])
    return Graph.hetero(nodes=node_payloads, edges=edges)


def _partition_ranges(graph: Graph, num_partitions: int) -> tuple[dict[str, int], list[dict[str, tuple[int, int]]]]:
    num_nodes_by_type = {node_type: graph._node_count(node_type) for node_type in graph.schema.node_types}
    max_nodes = max(num_nodes_by_type.values(), default=0)
    if num_partitions > max_nodes:
        raise ValueError("num_partitions must be <= the largest node-type cardinality")

    chunk_sizes = {
        node_type: (count + num_partitions - 1) // num_partitions if count > 0 else 0
        for node_type, count in num_nodes_by_type.items()
    }
    partition_ranges = []
    for partition_id in range(num_partitions):
        node_ranges = {}
        has_nodes = False
        for node_type in graph.schema.node_types:
            count = num_nodes_by_type[node_type]
            chunk_size = chunk_sizes[node_type]
            start = min(partition_id * chunk_size, count) if chunk_size > 0 else 0
            end = min(count, start + chunk_size) if chunk_size > 0 else 0
            node_ranges[node_type] = (start, end)
            has_nodes = has_nodes or start < end
        if has_nodes:
            partition_ranges.append(node_ranges)
    return num_nodes_by_type, partition_ranges


def _partition_subgraph(graph: Graph, node_ranges: dict[str, tuple[int, int]]) -> tuple[Graph, dict[str, torch.Tensor], dict]:
    first_edge_store = next(iter(graph.edges.values()), None)
    reference_device = first_edge_store.edge_index.device if first_edge_store is not None else torch.device("cpu")

    partition_nodes = {}
    node_ids_by_type = {}
    for node_type in graph.schema.node_types:
        start, end = node_ranges[node_type]
        node_data, node_ids = _slice_node_data(graph, node_type, start, end, device=reference_device)
        partition_nodes[node_type] = node_data
        node_ids_by_type[node_type] = node_ids

    partition_edges = {}
    boundary_edges = {}
    for edge_type, edge_store in graph.edges.items():
        src_type, _, dst_type = edge_type
        src_start, src_end = node_ranges[src_type]
        dst_start, dst_end = node_ranges[dst_type]
        edge_index = edge_store.edge_index
        src_mask = (edge_index[0] >= src_start) & (edge_index[0] < src_end)
        dst_mask = (edge_index[1] >= dst_start) & (edge_index[1] < dst_end)
        local_mask = src_mask & dst_mask
        boundary_mask = src_mask ^ dst_mask

        local_edge_index = edge_index[:, local_mask].clone()
        local_edge_index[0] -= src_start
        local_edge_index[1] -= dst_start
        partition_edges[edge_type] = _slice_edge_data(edge_store, local_mask, edge_index=local_edge_index)
        boundary_edges[edge_type] = _slice_edge_data(
            edge_store,
            boundary_mask,
            edge_index=edge_index[:, boundary_mask].clone(),
        )

    return _build_partition_graph(graph, partition_nodes, partition_edges), node_ids_by_type, boundary_edges


def write_partitioned_graph(graph: Graph, root, *, num_partitions: int) -> PartitionManifest:
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")

    num_nodes_by_type, partition_ranges = _partition_ranges(graph, num_partitions)

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    partitions = []
    single_node_type = len(graph.schema.node_types) == 1 and graph.schema.node_types == ("node",)
    for partition_id, node_ranges in enumerate(partition_ranges):
        shard_graph, node_ids_by_type, boundary_edges = _partition_subgraph(graph, node_ranges)
        filename = f"part-{partition_id}.pt"
        payload_node_ids = node_ids_by_type["node"] if single_node_type else node_ids_by_type
        node_feature_shapes = {
            str(node_type): {
                name: tuple(int(dim) for dim in value.shape)
                for name, value in shard_graph.nodes[str(node_type)].data.items()
                if isinstance(value, torch.Tensor)
            }
            for node_type in shard_graph.nodes
        }
        edge_ids_by_type = {
            edge_type: _tensor_int_tuple(shard_graph.edges[edge_type].data["e_id"])
            for edge_type in shard_graph.edges
        }
        edge_feature_shapes = {
            edge_type: {
                name: tuple(int(dim) for dim in value.shape)
                for name, value in shard_graph.edges[edge_type].data.items()
                if name != "edge_index" and isinstance(value, torch.Tensor)
            }
            for edge_type in shard_graph.edges
        }
        boundary_edge_ids_by_type = {
            edge_type: _tensor_int_tuple(boundary_edges[edge_type]["e_id"])
            for edge_type in shard_graph.edges
        }
        torch.save(
            {
                "partition_id": partition_id,
                "node_ids": payload_node_ids,
                "graph": serialize_graph(shard_graph),
                "boundary_edges": boundary_edges,
            },
            root / filename,
        )
        partitions.append(
            PartitionShard(
                partition_id=partition_id,
                node_range=node_ranges.get("node", (0, 0)),
                node_ranges=node_ranges,
                path=filename,
                metadata={
                    "node_feature_shapes": node_feature_shapes,
                    "edge_feature_shapes": edge_feature_shapes,
                    "edge_ids_by_type": edge_ids_by_type,
                    "boundary_edge_ids_by_type": boundary_edge_ids_by_type,
                },
            )
        )

    manifest = PartitionManifest(
        num_nodes=sum(num_nodes_by_type.values()),
        num_nodes_by_type=num_nodes_by_type,
        partitions=tuple(partitions),
        metadata={
            "num_edges": sum(int(store.edge_index.size(1)) for store in graph.edges.values()),
            "edge_types": tuple(tuple(edge_type) for edge_type in graph.edges),
            "time_attr": graph.schema.time_attr,
        },
    )
    save_partition_manifest(root / "manifest.json", manifest)
    return manifest


__all__ = ["write_partitioned_graph"]
