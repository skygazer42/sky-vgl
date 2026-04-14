import importlib.util
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_hotpaths.py"


def _load_benchmark_hotpaths_module(module_name: str = "benchmark_hotpaths"):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_benchmark_document_includes_schema_and_runner_metadata():
    benchmark_hotpaths = _load_benchmark_hotpaths_module("benchmark_hotpaths_schema")

    payload = benchmark_hotpaths.build_benchmark_document(
        preset="ci",
        config={"num_nodes": 5, "seed": 0},
        query_ops={"find_edges_seconds": 0.1},
        routing={"local_route_node_ids_seconds": 0.2},
        sampling={"link_neighbor_sample_seconds": 0.3},
        generated_at_utc="2026-04-14T00:00:00Z",
        runner={
            "python_version": "3.11.0",
            "python_implementation": "CPython",
            "platform": "Linux-1",
        },
    )

    assert payload["schema_version"] == benchmark_hotpaths.BENCHMARK_SCHEMA_VERSION
    assert payload["benchmark"] == "vgl_hotpaths"
    assert payload["generated_at_utc"] == "2026-04-14T00:00:00Z"
    assert payload["metric_unit"] == benchmark_hotpaths.BENCHMARK_METRIC_UNIT
    assert payload["runner"]["python_version"] == "3.11.0"
    assert payload["query_ops"]["find_edges_seconds"] == 0.1


def test_benchmark_document_is_json_serializable():
    benchmark_hotpaths = _load_benchmark_hotpaths_module("benchmark_hotpaths_json")

    payload = benchmark_hotpaths.build_benchmark_document(
        preset="smoke",
        config={"num_nodes": 1, "seed": 0},
        query_ops={},
        routing={},
        sampling={},
    )

    serialized = json.dumps(payload, sort_keys=True)

    assert '"schema_version"' in serialized
    assert '"runner"' in serialized
