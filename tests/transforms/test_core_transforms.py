import torch

from vgl import Graph
from vgl.dataloading import ListDataset
from vgl.transforms import (
    AddSelfLoops,
    Compose,
    FeatureStandardize,
    IdentityTransform,
    LargestConnectedComponents,
    NormalizeFeatures,
    RandomGraphSplit,
    RandomNodeSplit,
    RemoveSelfLoops,
    ToUndirected,
    TrainOnlyFeatureNormalizer,
)


def _base_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        x=torch.tensor(
            [
                [2.0, 0.0],
                [0.0, 4.0],
                [1.0, 1.0],
                [3.0, 3.0],
            ]
        ),
        y=torch.tensor([0, 0, 1, 1]),
    )


def _graph_without_x():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]]),
        y=torch.tensor([0, 0, 1, 1, 2]),
    )


def test_compose_applies_transforms_in_order():
    graph = _base_graph()

    transformed = Compose(
        [
            IdentityTransform(),
            NormalizeFeatures(),
            AddSelfLoops(),
        ]
    )(graph)

    assert torch.allclose(transformed.x.sum(dim=1), torch.ones(4))
    assert transformed.edge_index.size(1) == graph.edge_index.size(1) + graph.x.size(0)


def test_random_node_split_adds_boolean_masks():
    graph = _base_graph()

    transformed = RandomNodeSplit(num_val=1, num_test=1, seed=7)(graph)

    assert transformed.train_mask.dtype == torch.bool
    assert transformed.val_mask.dtype == torch.bool
    assert transformed.test_mask.dtype == torch.bool
    assert int(transformed.train_mask.sum()) == 2
    assert int(transformed.val_mask.sum()) == 1
    assert int(transformed.test_mask.sum()) == 1


def test_random_node_split_supports_class_balanced_train_selection():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]),
        x=torch.randn(6, 2),
        y=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    transformed = RandomNodeSplit(num_train_per_class=1, num_val=2, num_test=2, seed=3)(graph)

    train_labels = transformed.y[transformed.train_mask]
    assert train_labels.numel() == 2
    assert set(train_labels.tolist()) == {0, 1}


def test_random_node_split_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]),
        x=torch.randn(6, 2),
        y=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    def fail_tolist(self):
        raise AssertionError("RandomNodeSplit should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    transformed = RandomNodeSplit(num_train_per_class=1, num_val=2, num_test=2, seed=3)(graph)

    train_labels = transformed.y[transformed.train_mask]
    assert train_labels.numel() == 2
    assert torch.equal(torch.sort(train_labels).values, torch.tensor([0, 1]))


def test_random_node_split_avoids_tensor_item(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]),
        x=torch.randn(6, 2),
        y=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    def fail_item(self):
        raise AssertionError("RandomNodeSplit should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    transformed = RandomNodeSplit(num_train_per_class=1, num_val=2, num_test=2, seed=3)(graph)

    train_labels = transformed.y[transformed.train_mask]
    assert train_labels.numel() == 2
    assert torch.equal(torch.sort(train_labels).values, torch.tensor([0, 1]))


def test_random_node_split_avoids_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]),
        x=torch.randn(6, 2),
        y=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    def fail_int(self):
        raise AssertionError("RandomNodeSplit should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    transformed = RandomNodeSplit(num_train_per_class=1, num_val=2, num_test=2, seed=3)(graph)

    train_labels = transformed.y[transformed.train_mask]
    assert train_labels.numel() == 2
    assert torch.equal(torch.sort(train_labels).values, torch.tensor([0, 1]))


def test_random_node_split_accepts_tensor_seed_without_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]),
        x=torch.randn(6, 2),
        y=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    def fail_int(self):
        raise AssertionError("RandomNodeSplit should stay off tensor.__int__ for tensor seed values")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    transformed = RandomNodeSplit(num_train_per_class=1, num_val=2, num_test=2, seed=torch.tensor(3))(graph)

    train_labels = transformed.y[transformed.train_mask]
    assert train_labels.numel() == 2
    assert torch.equal(torch.sort(train_labels).values, torch.tensor([0, 1]))


def test_random_node_split_accepts_tensor_num_train_per_class_without_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]]),
        x=torch.randn(6, 2),
        y=torch.tensor([0, 0, 0, 1, 1, 1]),
    )

    def fail_int(self):
        raise AssertionError("RandomNodeSplit num_train_per_class should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    transformed = RandomNodeSplit(
        num_train_per_class=torch.tensor(1),
        num_val=2,
        num_test=2,
        seed=3,
    )(graph)

    train_labels = transformed.y[transformed.train_mask]
    assert train_labels.numel() == 2
    assert torch.equal(torch.sort(train_labels).values, torch.tensor([0, 1]))


def test_random_graph_split_returns_datasets():
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 2), y=torch.tensor([label]))
        for label in (0, 1, 0, 1, 0)
    ]

    train_ds, val_ds, test_ds = RandomGraphSplit(num_val=1, num_test=1, seed=11)(graphs)

    assert isinstance(train_ds, ListDataset)
    assert len(train_ds) == 3
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_random_graph_split_avoids_tensor_tolist(monkeypatch):
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 2), y=torch.tensor([label]))
        for label in (0, 1, 0, 1, 0)
    ]

    def fail_tolist(self):
        raise AssertionError("RandomGraphSplit should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    train_ds, val_ds, test_ds = RandomGraphSplit(num_val=1, num_test=1, seed=11)(graphs)

    assert isinstance(train_ds, ListDataset)
    assert len(train_ds) == 3
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_random_graph_split_avoids_tensor_item(monkeypatch):
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 2), y=torch.tensor([label]))
        for label in (0, 1, 0, 1, 0)
    ]

    def fail_item(self):
        raise AssertionError("RandomGraphSplit should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    train_ds, val_ds, test_ds = RandomGraphSplit(num_val=1, num_test=1, seed=11)(graphs)

    assert isinstance(train_ds, ListDataset)
    assert len(train_ds) == 3
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_random_graph_split_avoids_tensor_int(monkeypatch):
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 2), y=torch.tensor([label]))
        for label in (0, 1, 0, 1, 0)
    ]

    def fail_int(self):
        raise AssertionError("RandomGraphSplit should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    train_ds, val_ds, test_ds = RandomGraphSplit(num_val=1, num_test=1, seed=11)(graphs)

    assert isinstance(train_ds, ListDataset)
    assert len(train_ds) == 3
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_random_graph_split_accepts_tensor_seed_without_tensor_int(monkeypatch):
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 2), y=torch.tensor([label]))
        for label in (0, 1, 0, 1, 0)
    ]

    def fail_int(self):
        raise AssertionError("RandomGraphSplit should stay off tensor.__int__ for tensor seed values")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    train_ds, val_ds, test_ds = RandomGraphSplit(num_val=1, num_test=1, seed=torch.tensor(11))(graphs)

    assert isinstance(train_ds, ListDataset)
    assert len(train_ds) == 3
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_random_graph_split_accepts_tensor_count_parameters_without_tensor_int(monkeypatch):
    graphs = [
        Graph.homo(edge_index=torch.tensor([[0], [1]]), x=torch.randn(2, 2), y=torch.tensor([label]))
        for label in (0, 1, 0, 1, 0)
    ]

    def fail_int(self):
        raise AssertionError("RandomGraphSplit count parameters should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    train_ds, val_ds, test_ds = RandomGraphSplit(
        num_val=torch.tensor(1),
        num_test=torch.tensor(1),
        seed=11,
    )(graphs)

    assert isinstance(train_ds, ListDataset)
    assert len(train_ds) == 3
    assert len(val_ds) == 1
    assert len(test_ds) == 1


def test_normalize_features_row_normalizes_x():
    graph = _base_graph()

    transformed = NormalizeFeatures()(graph)

    assert torch.allclose(transformed.x.sum(dim=1), torch.ones(4))


def test_feature_standardize_uses_full_dataset_statistics():
    graph = _base_graph()

    transformed = FeatureStandardize()(graph)

    assert torch.allclose(transformed.x.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(transformed.x.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)


def test_train_only_feature_normalizer_uses_train_mask_statistics():
    graph = _base_graph()
    graph = RandomNodeSplit(num_val=1, num_test=1, seed=5)(graph)

    transformed = TrainOnlyFeatureNormalizer()(graph)
    train_x = transformed.x[transformed.train_mask]

    assert torch.allclose(train_x.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(train_x.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)


def test_train_only_feature_normalizer_avoids_tensor_int(monkeypatch):
    graph = _base_graph()
    graph = RandomNodeSplit(num_val=1, num_test=1, seed=5)(graph)

    def fail_int(self):
        raise AssertionError("TrainOnlyFeatureNormalizer should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    transformed = TrainOnlyFeatureNormalizer()(graph)
    train_x = transformed.x[transformed.train_mask]

    assert torch.allclose(train_x.mean(dim=0), torch.zeros(2), atol=1e-6)
    assert torch.allclose(train_x.std(dim=0, unbiased=False), torch.ones(2), atol=1e-6)


def test_to_undirected_mirrors_edges_and_edge_features():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
        edge_data={"weight": torch.tensor([0.5, 1.5])},
    )

    transformed = ToUndirected()(graph)
    edges = {tuple(edge.tolist()) for edge in transformed.edge_index.t()}

    assert edges == {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert transformed.edata["weight"].tolist() == [0.5, 1.5, 0.5, 1.5]


def test_to_undirected_transform_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
        edge_data={"weight": torch.tensor([0.5, 1.5])},
    )

    def fail_tolist(self):
        raise AssertionError("ToUndirected transform should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    transformed = ToUndirected()(graph)

    assert torch.equal(transformed.edge_index, torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]))


def test_to_undirected_transform_avoids_torch_unique(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 2),
        edge_data={"weight": torch.tensor([0.5, 1.5])},
    )

    def fail_unique(*args, **kwargs):
        raise AssertionError("ToUndirected transform should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    transformed = ToUndirected()(graph)

    assert torch.equal(transformed.edge_index, torch.tensor([[0, 1, 1, 2], [1, 2, 0, 1]]))


def test_to_undirected_supports_homo_graph_without_x():
    transformed = ToUndirected()(_graph_without_x())

    assert transformed._node_count("node") == 5
    assert {tuple(edge.tolist()) for edge in transformed.edge_index.t()} == {
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (3, 4),
        (4, 3),
    }


def test_to_undirected_assigns_unique_ids_to_synthesized_reverse_edges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(2, 2),
        edge_data={"e_id": torch.tensor([42])},
    )

    transformed = ToUndirected()(graph)

    assert torch.equal(transformed.edata["e_id"], torch.tensor([42, 43]))


def test_add_and_remove_self_loops_round_trip():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 3),
        edge_data={"weight": torch.tensor([2.0, 3.0])},
    )

    with_loops = AddSelfLoops(fill_value=1.25)(graph)
    without_loops = RemoveSelfLoops()(with_loops)

    loop_edges = {tuple(edge.tolist()) for edge in with_loops.edge_index.t()}
    assert (0, 0) in loop_edges
    assert (1, 1) in loop_edges
    assert {tuple(edge.tolist()) for edge in without_loops.edge_index.t()} == {(0, 1), (1, 0)}


def test_add_self_loops_transform_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(3, 3),
        edge_data={"weight": torch.tensor([2.0, 3.0])},
    )

    def fail_tolist(self):
        raise AssertionError("AddSelfLoops transform should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    transformed = AddSelfLoops(fill_value=1.25)(graph)

    assert torch.equal(transformed.edge_index, torch.tensor([[0, 1, 0, 1, 2], [1, 0, 0, 1, 2]]))


def test_add_self_loops_supports_homo_graph_without_x():
    transformed = AddSelfLoops()(_graph_without_x())

    assert transformed._node_count("node") == 5
    assert {(int(src), int(dst)) for src, dst in transformed.edge_index.t()} >= {
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
    }


def test_add_self_loops_assigns_unique_ids_to_synthesized_loop_edges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.randn(3, 2),
        edge_data={"e_id": torch.tensor([42])},
    )

    transformed = AddSelfLoops()(graph)

    assert torch.equal(transformed.edata["e_id"], torch.tensor([42, 43, 44, 45]))


def test_largest_connected_components_keeps_biggest_component():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]]),
        x=torch.arange(10, dtype=torch.float32).view(5, 2),
    )

    transformed = LargestConnectedComponents()(graph)

    assert transformed.x.size(0) == 3
    assert {tuple(edge.tolist()) for edge in transformed.edge_index.t()} == {(0, 1), (1, 2)}


def test_largest_connected_components_avoids_tensor_tolist(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]]),
        x=torch.arange(10, dtype=torch.float32).view(5, 2),
    )

    def fail_tolist(self):
        raise AssertionError("LargestConnectedComponents should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    transformed = LargestConnectedComponents()(graph)

    assert transformed.x.size(0) == 3
    assert torch.equal(transformed.edge_index, torch.tensor([[0, 1], [1, 2]]))


def test_largest_connected_components_avoids_tensor_item(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]]),
        x=torch.arange(10, dtype=torch.float32).view(5, 2),
    )

    def fail_item(self):
        raise AssertionError("LargestConnectedComponents should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    transformed = LargestConnectedComponents()(graph)

    assert transformed.x.size(0) == 3
    assert torch.equal(transformed.edge_index, torch.tensor([[0, 1], [1, 2]]))


def test_largest_connected_components_avoids_python_deque(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]]),
        x=torch.arange(10, dtype=torch.float32).view(5, 2),
    )

    import vgl.transforms.structure as structure_transforms

    def fail_deque(*args, **kwargs):
        raise AssertionError("LargestConnectedComponents should avoid Python deque traversal")

    monkeypatch.setattr(structure_transforms, "deque", fail_deque, raising=False)

    transformed = LargestConnectedComponents()(graph)

    assert transformed.x.size(0) == 3
    assert torch.equal(transformed.edge_index, torch.tensor([[0, 1], [1, 2]]))


def test_largest_connected_components_accepts_tensor_num_components_without_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 4]]),
        x=torch.arange(10, dtype=torch.float32).view(5, 2),
    )

    def fail_int(self):
        raise AssertionError("LargestConnectedComponents num_components should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    transformed = LargestConnectedComponents(num_components=torch.tensor(1))(graph)

    assert transformed.x.size(0) == 3
    assert torch.equal(transformed.edge_index, torch.tensor([[0, 1], [1, 2]]))


def test_largest_connected_components_supports_homo_graph_without_x():
    transformed = LargestConnectedComponents()(_graph_without_x())

    assert transformed._node_count("node") == 3
    assert torch.equal(transformed.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(transformed.y, torch.tensor([0, 0, 1]))
