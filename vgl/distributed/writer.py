from pathlib import Path

import torch

from vgl.data.ondisk import serialize_graph
from vgl.distributed.partition import PartitionManifest, PartitionShard, save_partition_manifest
from vgl.graph.graph import Graph


def _partition_subgraph(graph: Graph, start: int, end: int) -> Graph:
    edge_type = graph._default_edge_type()
    edge_store = graph.edges[edge_type]
    edge_index = edge_store.edge_index
    edge_mask = (
        (edge_index[0] >= start)
        & (edge_index[0] < end)
        & (edge_index[1] >= start)
        & (edge_index[1] < end)
    )
    local_edge_index = edge_index[:, edge_mask] - start
    node_count = graph._node_count("node")
    node_data = {}
    for key, value in graph.nodes["node"].data.items():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == node_count:
            node_data[key] = value[start:end]
        else:
            node_data[key] = value
    node_data.setdefault("n_id", torch.arange(start, end, dtype=torch.long, device=edge_index.device))

    edge_count = int(edge_index.size(1))
    edge_data = {}
    for key, value in edge_store.data.items():
        if key == "edge_index":
            continue
        if isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[edge_mask]
        else:
            edge_data[key] = value
    edge_data.setdefault(
        "e_id",
        torch.arange(edge_count, dtype=torch.long, device=edge_index.device)[edge_mask],
    )
    return Graph.homo(edge_index=local_edge_index, edge_data=edge_data, **node_data)


def write_partitioned_graph(graph: Graph, root, *, num_partitions: int) -> PartitionManifest:
    if num_partitions < 1:
        raise ValueError("num_partitions must be >= 1")
    if set(graph.nodes) != {"node"} or len(graph.edges) != 1 or graph.schema.time_attr is not None:
        raise ValueError("write_partitioned_graph currently supports homogeneous non-temporal graphs only")

    num_nodes = graph._node_count("node")
    if num_partitions > num_nodes:
        raise ValueError("num_partitions must be <= num_nodes")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    chunk_size = (num_nodes + num_partitions - 1) // num_partitions
    partitions = []
    for partition_id in range(num_partitions):
        start = partition_id * chunk_size
        end = min(num_nodes, start + chunk_size)
        if start >= end:
            break
        shard_graph = _partition_subgraph(graph, start, end)
        filename = f"part-{partition_id}.pt"
        torch.save(
            {
                "partition_id": partition_id,
                "node_ids": torch.arange(start, end, dtype=torch.long),
                "graph": serialize_graph(shard_graph),
            },
            root / filename,
        )
        partitions.append(
            PartitionShard(
                partition_id=partition_id,
                node_range=(start, end),
                path=filename,
            )
        )

    manifest = PartitionManifest(
        num_nodes=num_nodes,
        partitions=tuple(partitions),
        metadata={"num_edges": int(graph.edge_index.size(1))},
    )
    save_partition_manifest(root / "manifest.json", manifest)
    return manifest


__all__ = ["write_partitioned_graph"]
