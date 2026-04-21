from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_distributed_docs_use_current_store_bootstrap_signatures():
    api_distributed = _read("docs/api/distributed.md")
    example_distributed = _read("docs/examples/distributed-partition.md")
    sampling_guide = _read("docs/guide/sampling.md")
    architecture = _read("docs/architecture.md")

    for text in (api_distributed, example_distributed, sampling_guide):
        assert 'StoreBackedSamplingCoordinator.from_partition_dir("artifacts/partitions")' in text
        assert 'manifest, feature_store, graph_store = load_partitioned_stores("artifacts/partitions")' in text
        assert 'graph_store, feature_store = load_partitioned_stores("artifacts/partitions")' not in text
        assert "StoreBackedSamplingCoordinator(graph_store, feature_store)" not in text

    assert "`load_partitioned_stores(dir)` 返回 `(manifest, PartitionedFeatureStore, PartitionedGraphStore)`" in architecture
    assert "`load_partitioned_stores(dir)` 返回 `(PartitionedGraphStore, PartitionedFeatureStore)`" not in architecture


def test_distributed_partition_example_matches_current_partition_artifact_layout():
    example_distributed = _read("docs/examples/distributed-partition.md")

    assert "manifest.json" in example_distributed
    assert "part-0.pt" in example_distributed
    assert "part-1.pt" in example_distributed
    assert "part-000/" not in example_distributed
    assert "edges.bin" not in example_distributed
    assert "features/" not in example_distributed
    assert "node_node_x.bin" not in example_distributed


def test_interop_docs_use_current_csv_helper_names():
    compat_api = _read("docs/api/compat.md")
    interop_example = _read("docs/examples/interop.md")
    graph_guide = _read("docs/guide/graph.md")
    sparse_api = _read("docs/api/sparse.md")

    for text in (compat_api, interop_example):
        assert 'Graph.from_edge_list_csv("edges.csv")' in text
        assert 'Graph.from_csv_tables("nodes.csv", "edges.csv")' in text
        assert "Graph.from_csv_edge_list(" not in text
        assert 'Graph.from_csv("nodes.csv", "edges.csv")' not in text

    for text in (interop_example, graph_guide):
        assert "to_networkx(graph, node_attrs=" not in text
        assert "from_networkx(nx_graph, node_type=" not in text
        assert "to_torch_sparse(vst, layout=" not in text
        assert "vst2.to_coo().indices()" not in text
        assert "coo = to_coo(vst2)" in text
        assert "edge_index = torch.stack((coo.row, coo.col))" in text

    assert "back = to_torch_sparse(vst)" in sparse_api
    assert "to_torch_sparse(vst, layout=" not in sparse_api


def test_networkx_docs_describe_homogeneous_only_support():
    graph_guide = _read("docs/guide/graph.md")
    interop_example = _read("docs/examples/interop.md")

    for text in (graph_guide, interop_example):
        assert "同构图" in text
        assert "单关系异构图" not in text
        assert "异构图会保留 `ntype` / `etype` 键" not in text


def test_ondisk_data_docs_and_changelog_match_current_behavior():
    api_data = _read("docs/api/data.md")
    changelog = _read("docs/changelog.md")

    assert "graph-000000.pt" in api_data
    assert "按需逐条 `torch.load(...)`" in api_data
    assert "manifest-backed split view" in api_data
    assert "旧版 `graphs.pt` 数据集仍可读取" in api_data
    assert "DatasetManifest.feature_shapes" not in api_data
    assert "lazy edge_index" not in api_data

    assert "OnDiskGraphDataset` 的逐图 payload 布局" in changelog
    assert "DatasetManifest.feature_shapes" not in changelog
    assert "lazy edge_index" not in changelog


def test_migration_guide_warning_type_matches_runtime_contract():
    migration_guide = _read("docs/migration-guide.md")

    assert "FutureWarning" in migration_guide
    assert "DeprecationWarning" not in migration_guide
