from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
import repo_script_imports

ensure_repo_root_on_path = repo_script_imports.ensure_repo_root_on_path


ensure_repo_root_on_path()

BENCHMARK_SCHEMA_VERSION = 1
BENCHMARK_METRIC_UNIT = "seconds"
PRESETS = {
    "smoke": {
        "num_nodes": 100,
        "num_edges": 500,
        "num_queries": 25,
        "num_partitions": 2,
        "warmup": 1,
        "repeats": 1,
    },
    "ci": {
        "num_nodes": 5_000,
        "num_edges": 20_000,
        "num_queries": 1_000,
        "num_partitions": 4,
        "warmup": 2,
        "repeats": 3,
    },
    "default": {
        "num_nodes": 200_000,
        "num_edges": 1_000_000,
        "num_queries": 20_000,
        "num_partitions": 8,
        "warmup": 3,
        "repeats": 10,
    },
}


def build_benchmark_document(
    *,
    preset: str,
    config: dict[str, int],
    query_ops: dict[str, float],
    routing: dict[str, float],
    sampling: dict[str, float],
    generated_at_utc: str | None = None,
    runner: dict[str, str] | None = None,
) -> dict[str, object]:
    if generated_at_utc is None:
        generated_at_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if runner is None:
        runner = {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(),
        }
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "benchmark": "vgl_hotpaths",
        "generated_at_utc": generated_at_utc,
        "preset": preset,
        "config": dict(config),
        "runner": dict(runner),
        "metric_unit": BENCHMARK_METRIC_UNIT,
        "query_ops": dict(query_ops),
        "routing": dict(routing),
        "sampling": dict(sampling),
    }


def _time_call(fn, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    start = perf_counter()
    for _ in range(repeats):
        fn()
    end = perf_counter()
    return (end - start) / max(repeats, 1)


def _build_graph(*, num_nodes: int, num_edges: int, seed: int):
    import torch
    from vgl.graph import Graph

    generator = torch.Generator().manual_seed(seed)
    src = torch.randint(num_nodes, (num_edges,), generator=generator)
    dst = torch.randint(num_nodes, (num_edges,), generator=generator)
    edge_index = torch.stack((src, dst))
    x = torch.randn(num_nodes, 64, generator=generator)
    return Graph.homo(edge_index=edge_index, x=x)


def benchmark_query_ops(*, num_nodes: int, num_edges: int, num_queries: int, warmup: int, repeats: int, seed: int):
    import torch
    from vgl.ops import edge_ids, find_edges, has_edges_between

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
    import torch
    from vgl.distributed.coordinator import LocalSamplingCoordinator, StoreBackedSamplingCoordinator
    from vgl.distributed.shard import LocalGraphShard
    from vgl.distributed.writer import write_partitioned_graph

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
    import torch
    from vgl.dataloading import LinkNeighborSampler, TemporalNeighborSampler
    from vgl.data.sample import LinkPredictionRecord, TemporalEventRecord
    from vgl.graph import Graph

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


def _dump_results(results: dict[str, object], *, output: Path | None, emit_stdout: bool) -> None:
    document = json.dumps(results, indent=2, sort_keys=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(document, encoding="utf-8")
    if emit_stdout or output is None:
        print(document)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark VGL single-machine query and routing hotpaths.")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="default")
    parser.add_argument("--num-nodes", type=int)
    parser.add_argument("--num-edges", type=int)
    parser.add_argument("--num-queries", type=int)
    parser.add_argument("--num-partitions", type=int)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repeats", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Write JSON benchmark results to this file")
    parser.add_argument("--print", action="store_true", dest="emit_stdout", help="Also print JSON to stdout")
    args = parser.parse_args()
    config = dict(PRESETS[args.preset])
    overrides = {
        "num_nodes": args.num_nodes,
        "num_edges": args.num_edges,
        "num_queries": args.num_queries,
        "num_partitions": args.num_partitions,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "seed": args.seed,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})

    benchmark_config = {
        "num_nodes": config["num_nodes"],
        "num_edges": config["num_edges"],
        "num_queries": config["num_queries"],
        "num_partitions": config["num_partitions"],
        "warmup": config["warmup"],
        "repeats": config["repeats"],
        "seed": config["seed"],
    }
    results = build_benchmark_document(
        preset=args.preset,
        config=benchmark_config,
        query_ops=benchmark_query_ops(
            num_nodes=config["num_nodes"],
            num_edges=config["num_edges"],
            num_queries=config["num_queries"],
            warmup=config["warmup"],
            repeats=config["repeats"],
            seed=config["seed"],
        ),
        routing=benchmark_routing(
            num_nodes=config["num_nodes"],
            num_edges=config["num_edges"],
            num_partitions=config["num_partitions"],
            num_queries=config["num_queries"],
            warmup=config["warmup"],
            repeats=config["repeats"],
            seed=config["seed"],
        ),
        sampling=benchmark_sampling(
            num_nodes=config["num_nodes"],
            num_edges=config["num_edges"],
            warmup=config["warmup"],
            repeats=config["repeats"],
            seed=config["seed"],
        ),
    )
    _dump_results(results, output=args.output, emit_stdout=args.emit_stdout)


if __name__ == "__main__":
    main()
