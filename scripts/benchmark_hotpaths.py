from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import torch

from vgl import Graph
from vgl.data.sample import LinkPredictionRecord, TemporalEventRecord
from vgl.data.sampler import LinkNeighborSampler, TemporalNeighborSampler
from vgl.distributed.coordinator import LocalSamplingCoordinator, StoreBackedSamplingCoordinator
from vgl.distributed.shard import LocalGraphShard
from vgl.distributed.writer import write_partitioned_graph
from vgl.ops import edge_ids, find_edges, has_edges_between


def _time_call(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    start = perf_counter()
    for _ in range(repeats):
        fn()
    end = perf_counter()
    return (end - start) / max(repeats, 1)


def _build_graph(*, num_nodes: int, num_edges: int, seed: int) -> Graph:
    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(num_nodes, (num_edges,), generator=generator)
    dst = torch.randint(num_nodes, (num_edges,), generator=generator)
    edge_index = torch.stack((src, dst))
    x = torch.randn(num_nodes, 64, generator=generator)
    return Graph.homo(edge_index=edge_index, x=x)


def benchmark_query_ops(*, num_nodes: int, num_edges: int, num_queries: int, warmup: int, repeats: int, seed: int):
    graph = _build_graph(num_nodes=num_nodes, num_edges=num_edges, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    query_eids = torch.randint(num_edges, (num_queries,), generator=generator)
    query_src = graph.edge_index[0, query_eids]
    query_dst = graph.edge_index[1, query_eids]

    return {
        "find_edges_seconds": _time_call(
            lambda: find_edges(graph, query_eids),
            warmup=warmup,
            repeats=repeats,
        ),
        "edge_ids_seconds": _time_call(
            lambda: edge_ids(graph, query_src, query_dst),
            warmup=warmup,
            repeats=repeats,
        ),
        "has_edges_between_seconds": _time_call(
            lambda: has_edges_between(graph, query_src, query_dst),
            warmup=warmup,
            repeats=repeats,
        ),
    }


def benchmark_routing(*, num_nodes: int, num_edges: int, num_partitions: int, num_queries: int, warmup: int, repeats: int, seed: int):
    graph = _build_graph(num_nodes=num_nodes, num_edges=num_edges, seed=seed)
    generator = torch.Generator().manual_seed(seed + 2)
    query_nodes = torch.randint(num_nodes, (num_queries,), generator=generator)

    with TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        write_partitioned_graph(graph, root, num_partitions=num_partitions)
        local = LocalSamplingCoordinator(
            {
                partition_id: LocalGraphShard.from_partition_dir(root, partition_id=partition_id)
                for partition_id in range(num_partitions)
            }
        )
        store_backed = StoreBackedSamplingCoordinator.from_partition_dir(root)
        return {
            "local_route_node_ids_seconds": _time_call(
                lambda: local.route_node_ids(query_nodes),
                warmup=warmup,
                repeats=repeats,
            ),
            "store_backed_route_node_ids_seconds": _time_call(
                lambda: store_backed.route_node_ids(query_nodes),
                warmup=warmup,
                repeats=repeats,
            ),
        }


def benchmark_sampling(*, num_nodes: int, num_edges: int, warmup: int, repeats: int, seed: int):
    graph = _build_graph(num_nodes=num_nodes, num_edges=num_edges, seed=seed)
    link_sampler = LinkNeighborSampler(num_neighbors=[-1, -1], seed=seed)
    link_record = LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1)

    temporal_graph = Graph.temporal(
        nodes={"node": {"x": graph.x}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": graph.edge_index,
                "timestamp": torch.arange(num_edges, dtype=torch.long),
            }
        },
        time_attr="timestamp",
    )
    temporal_sampler = TemporalNeighborSampler(num_neighbors=[-1, -1], seed=seed)
    temporal_record = TemporalEventRecord(
        graph=temporal_graph,
        src_index=0,
        dst_index=1,
        timestamp=max(num_edges - 1, 0),
        label=1,
    )

    return {
        "link_neighbor_sample_seconds": _time_call(
            lambda: link_sampler.sample(link_record),
            warmup=warmup,
            repeats=repeats,
        ),
        "temporal_neighbor_sample_seconds": _time_call(
            lambda: temporal_sampler.sample(temporal_record),
            warmup=warmup,
            repeats=repeats,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VGL single-machine query and routing hotpaths.")
    parser.add_argument("--num-nodes", type=int, default=200_000)
    parser.add_argument("--num-edges", type=int, default=1_000_000)
    parser.add_argument("--num-queries", type=int, default=20_000)
    parser.add_argument("--num-partitions", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = {
        "config": {
            "num_nodes": args.num_nodes,
            "num_edges": args.num_edges,
            "num_queries": args.num_queries,
            "num_partitions": args.num_partitions,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "seed": args.seed,
        },
        "query_ops": benchmark_query_ops(
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            num_queries=args.num_queries,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        ),
        "routing": benchmark_routing(
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            num_partitions=args.num_partitions,
            num_queries=args.num_queries,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        ),
        "sampling": benchmark_sampling(
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            warmup=args.warmup,
            repeats=args.repeats,
            seed=args.seed,
        ),
    }
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
