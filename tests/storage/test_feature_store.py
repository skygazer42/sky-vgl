import pytest
import torch

from vgl.storage import FeatureStore, InMemoryTensorStore


NODE_X = ("node", "paper", "x")
EDGE_W = ("edge", ("paper", "cites", "paper"), "weight")


def test_feature_store_fetches_from_named_tensor_stores():
    store = FeatureStore(
        {
            NODE_X: InMemoryTensorStore(torch.tensor([[1.0], [2.0], [3.0]])),
            EDGE_W: InMemoryTensorStore(torch.tensor([[0.1], [0.2], [0.3], [0.4]])),
        }
    )

    result = store.fetch(NODE_X, torch.tensor([2, 0]))

    assert result.index.tolist() == [2, 0]
    assert torch.equal(result.values, torch.tensor([[3.0], [1.0]]))
    assert store.shape(NODE_X) == (3, 1)


def test_feature_store_raises_for_unknown_key():
    store = FeatureStore({})

    with pytest.raises(KeyError, match="unknown feature key"):
        store.fetch(NODE_X, torch.tensor([0]))
