import torch

from vgl.distributed.store import LocalFeatureStoreAdapter, LocalGraphStoreAdapter
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
    assert torch.equal(adapter.edge_index(partition_id=3), torch.tensor([[0, 1], [1, 2]]))
    assert adapter.edge_count(partition_id=3) == 2
    assert adjacency.shape == (3, 3)
