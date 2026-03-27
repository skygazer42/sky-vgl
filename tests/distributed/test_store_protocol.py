import torch

from vgl import Graph
import vgl.distributed.store as distributed_store_module
from vgl.distributed.store import (
    LocalFeatureStoreAdapter,
    LocalGraphStoreAdapter,
    PartitionedFeatureStore,
    PartitionedGraphStore,
    load_partitioned_stores,
)
from vgl.distributed.writer import write_partitioned_graph
from vgl.storage import FeatureStore, InMemoryGraphStore, InMemoryTensorStore


NODE_KEY = ("node", "node", "x")


def test_local_feature_store_adapter_forwards_fetch_and_shape():
    store = FeatureStore({NODE_KEY: InMemoryTensorStore(torch.tensor([[1.0], [2.0], [3.0]]))})
    adapter = LocalFeatureStoreAdapter(store)

    fetched = adapter.fetch(NODE_KEY, torch.tensor([2, 0]), partition_id=7)

    assert adapter.shape(NODE_KEY) == (3, 1)
    assert torch.equal(fetched.index, torch.tensor([2, 0]))
    assert torch.equal(fetched.values, torch.tensor([[3.0], [1.0]]))


def test_local_graph_store_adapter_forwards_graph_queries():
    backend = InMemoryGraphStore(
        edges={("node", "to", "node"): torch.tensor([[0, 1], [1, 2]])},
        num_nodes={"node": 3},
    )
    adapter = LocalGraphStoreAdapter(backend)

    adjacency = adapter.adjacency(partition_id=3)

    assert adapter.edge_types == (("node", "to", "node"),)
    assert adapter.num_nodes(partition_id=3) == 3
    assert torch.equal(adapter.edge_index(partition_id=3), torch.tensor([[0, 1], [1, 2]]))
    assert adapter.edge_count(partition_id=3) == 2
    assert adjacency.shape == (3, 3)


def test_partitioned_feature_store_dispatches_partition_specific_fetches():
    part0 = LocalFeatureStoreAdapter(
        FeatureStore({NODE_KEY: InMemoryTensorStore(torch.tensor([[1.0], [2.0]]))})
    )
    part1 = LocalFeatureStoreAdapter(
        FeatureStore({NODE_KEY: InMemoryTensorStore(torch.tensor([[3.0], [4.0]]))})
    )
    store = PartitionedFeatureStore({0: part0, 1: part1})

    fetched = store.fetch(NODE_KEY, torch.tensor([1, 0]), partition_id=1)

    assert store.shape(NODE_KEY) == (2, 1)
    assert torch.equal(fetched.index, torch.tensor([1, 0]))
    assert torch.equal(fetched.values, torch.tensor([[4.0], [3.0]]))


def test_local_feature_store_adapter_fetches_boundary_edge_features():
    edge_type = ("node", "to", "node")
    key = ("edge", edge_type, "weight")
    adapter = LocalFeatureStoreAdapter(
        FeatureStore({}),
        boundary_edge_data_by_type={
            edge_type: {
                "e_id": torch.tensor([4, 7]),
                "weight": torch.tensor([1.5, 2.5]),
            }
        },
    )

    fetched = adapter.fetch_boundary(key, torch.tensor([7, 4]), partition_id=0)

    assert torch.equal(fetched.index, torch.tensor([7, 4]))
    assert torch.equal(fetched.values, torch.tensor([2.5, 1.5]))


def test_partitioned_graph_store_dispatches_local_and_boundary_edge_queries():
    edge_type = ("node", "to", "node")
    part0 = LocalGraphStoreAdapter(
        InMemoryGraphStore(
            edges={edge_type: torch.tensor([[0], [1]])},
            num_nodes={"node": 2},
        ),
        boundary_edge_index_by_type={edge_type: torch.tensor([[1], [2]])},
    )
    part1 = LocalGraphStoreAdapter(
        InMemoryGraphStore(
            edges={edge_type: torch.tensor([[0], [1]])},
            num_nodes={"node": 2},
        ),
        boundary_edge_index_by_type={edge_type: torch.tensor([[0], [3]])},
    )
    store = PartitionedGraphStore({0: part0, 1: part1})

    adjacency = store.adjacency(partition_id=0)

    assert store.edge_types == (edge_type,)
    assert store.num_nodes(partition_id=0) == 2
    assert torch.equal(store.edge_index(partition_id=1), torch.tensor([[0], [1]]))
    assert torch.equal(store.boundary_edge_index(partition_id=0), torch.tensor([[1], [2]]))
    assert store.edge_count(partition_id=1) == 1
    assert adjacency.shape == (2, 2)


def test_load_partitioned_stores_builds_partition_aware_stores_from_partition_directory(tmp_path):
    edge_type = ("node", "to", "node")
    weight_key = ("edge", edge_type, "weight")
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
        edge_data={"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    manifest, feature_store, graph_store = load_partitioned_stores(tmp_path)

    fetched_node = feature_store.fetch(NODE_KEY, torch.tensor([1, 0]), partition_id=1)
    fetched_boundary = feature_store.fetch_boundary(weight_key, torch.tensor([1, 3]), partition_id=0)
    local_edge_index = graph_store.edge_index(partition_id=1)
    boundary_edge_index = graph_store.boundary_edge_index(partition_id=0)

    assert manifest.num_partitions == 2
    assert torch.equal(fetched_node.index, torch.tensor([1, 0]))
    assert torch.equal(fetched_node.values, torch.tensor([[6.0, 7.0], [4.0, 5.0]]))
    assert torch.equal(fetched_boundary.index, torch.tensor([1, 3]))
    assert torch.equal(fetched_boundary.values, torch.tensor([2.0, 4.0]))
    assert torch.equal(local_edge_index, torch.tensor([[0], [1]]))
    assert torch.equal(boundary_edge_index, torch.tensor([[1, 3], [2, 0]]))


def test_load_partitioned_stores_lazily_loads_and_reuses_partition_payloads(monkeypatch, tmp_path):
    edge_type = ("node", "to", "node")
    weight_key = ("edge", edge_type, "weight")
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        x=torch.arange(8, dtype=torch.float32).view(4, 2),
        edge_data={"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )
    write_partitioned_graph(graph, tmp_path, num_partitions=2)

    load_calls = []
    real_torch_load = distributed_store_module.torch.load

    def counting_load(path, *args, **kwargs):
        load_calls.append(str(path))
        return real_torch_load(path, *args, **kwargs)

    monkeypatch.setattr(distributed_store_module.torch, "load", counting_load)

    _, feature_store, graph_store = load_partitioned_stores(tmp_path)

    assert load_calls == []
    assert graph_store.edge_types == (edge_type,)
    assert graph_store.num_nodes(partition_id=1) == 2
    assert load_calls == []

    feature_store.fetch(NODE_KEY, torch.tensor([1, 0]), partition_id=1)
    assert len(load_calls) == 1
    assert load_calls[0].endswith("part-1.pt")

    graph_store.edge_index(partition_id=1)
    assert len(load_calls) == 1

    feature_store.fetch_boundary(weight_key, torch.tensor([1, 3]), partition_id=0)
    assert len(load_calls) == 2
    assert load_calls[1].endswith("part-0.pt")
