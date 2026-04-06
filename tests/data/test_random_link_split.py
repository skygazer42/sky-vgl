import inspect
import torch

from vgl import Graph
from vgl.data.sample import LinkPredictionRecord
from vgl.transforms import RandomLinkSplit


def _record_edges(records):
    return {(int(record.src_index), int(record.dst_index)) for record in records}


def test_random_link_split_returns_train_val_test_link_prediction_datasets():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(num_val=1, num_test=1, seed=7)(graph)

    train_records = list(train_dataset)
    val_records = list(val_dataset)
    test_records = list(test_dataset)

    assert len(train_records) == 3
    assert len(val_records) == 1
    assert len(test_records) == 1
    assert all(isinstance(record, LinkPredictionRecord) for record in train_records + val_records + test_records)
    assert {record.metadata["split"] for record in train_records} == {"train"}
    assert {record.metadata["split"] for record in val_records} == {"val"}
    assert {record.metadata["split"] for record in test_records} == {"test"}


def test_random_link_split_uses_train_graph_for_validation_and_train_plus_val_for_test():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(num_val=1, num_test=1, seed=0)(graph)

    train_records = list(train_dataset)
    val_records = list(val_dataset)
    test_records = list(test_dataset)
    train_edges = _record_edges(train_records)
    val_edges = _record_edges(val_records)

    assert _record_edges(train_records) == {
        tuple(edge.tolist())
        for edge in train_records[0].graph.edge_index.t()
    }
    assert {
        tuple(edge.tolist())
        for edge in val_records[0].graph.edge_index.t()
    } == train_edges
    assert {
        tuple(edge.tolist())
        for edge in test_records[0].graph.edge_index.t()
    } == train_edges | val_edges


def test_random_link_split_accepts_tensor_split_counts_without_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )

    def fail_int(self):
        raise AssertionError("RandomLinkSplit split counts should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=torch.tensor(1),
        num_test=torch.tensor(1),
        seed=7,
    )(graph)

    assert len(list(train_dataset)) == 3
    assert len(list(val_dataset)) == 1
    assert len(list(test_dataset)) == 1


def test_random_link_split_keeps_reverse_edges_in_same_split_when_undirected():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        x=torch.randn(4, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        is_undirected=True,
        seed=3,
    )(graph)

    split_edges = {
        "train": _record_edges(list(train_dataset)),
        "val": _record_edges(list(val_dataset)),
        "test": _record_edges(list(test_dataset)),
    }

    for src_index, dst_index in graph.edge_index.t().tolist():
        reverse = (int(dst_index), int(src_index))
        containing_splits = [
            split_name
            for split_name, edges in split_edges.items()
            if (int(src_index), int(dst_index)) in edges or reverse in edges
        ]
        assert len(set(containing_splits)) == 1


def test_random_link_split_supports_hetero_edge_type_and_reverse_edge_type():
    edge_type = ("author", "writes", "paper")
    reverse_edge_type = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1, 2, 1], [1, 2, 3, 0]])},
            reverse_edge_type: {"edge_index": torch.tensor([[1, 2, 3, 0], [0, 1, 2, 1]])},
        },
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        edge_type=edge_type,
        rev_edge_type=reverse_edge_type,
        seed=13,
    )(graph)

    train_records = list(train_dataset)
    val_records = list(val_dataset)
    test_records = list(test_dataset)

    assert len(train_records) == 2
    assert len(val_records) == 1
    assert len(test_records) == 1
    assert all(record.edge_type == edge_type for record in train_records + val_records + test_records)
    assert all(record.reverse_edge_type == reverse_edge_type for record in train_records + val_records + test_records)
    assert {
        tuple(edge.tolist())
        for edge in val_records[0].graph.edges[edge_type].edge_index.t()
    } == _record_edges(train_records)

    train_edges = _record_edges(train_records)
    val_edges = _record_edges(val_records)
    expected_test_graph_edges = train_edges | val_edges
    actual_test_graph_edges = {
        tuple(edge.tolist())
        for edge in test_records[0].graph.edges[edge_type].edge_index.t()
    }
    assert actual_test_graph_edges == expected_test_graph_edges

    reverse_graph_edges = {
        tuple(edge.tolist())
        for edge in test_records[0].graph.edges[reverse_edge_type].edge_index.t()
    }
    assert reverse_graph_edges == {(dst, src) for src, dst in expected_test_graph_edges}


def test_random_link_split_supports_disjoint_train_ratio():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
        x=torch.randn(3, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        disjoint_train_ratio=0.5,
        seed=5,
    )(graph)

    train_records = list(train_dataset)
    val_records = list(val_dataset)
    test_records = list(test_dataset)

    train_supervision_edges = _record_edges(train_records)
    train_message_passing_edges = {
        tuple(edge.tolist())
        for edge in train_records[0].graph.edge_index.t()
    }
    val_message_passing_edges = {
        tuple(edge.tolist())
        for edge in val_records[0].graph.edge_index.t()
    }
    test_message_passing_edges = {
        tuple(edge.tolist())
        for edge in test_records[0].graph.edge_index.t()
    }

    assert len(train_records) == 2
    assert len(train_message_passing_edges) == 2
    assert train_supervision_edges.isdisjoint(train_message_passing_edges)
    assert val_message_passing_edges == train_message_passing_edges
    assert test_message_passing_edges == train_message_passing_edges | _record_edges(val_records)


def test_random_link_split_can_add_negative_samples():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )
    positive_edges = {
        (int(src_index), int(dst_index))
        for src_index, dst_index in graph.edge_index.t().tolist()
    }

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        seed=9,
    )(graph)

    for dataset in (train_dataset, val_dataset, test_dataset):
        records = list(dataset)
        labels = [int(record.label) for record in records]
        negatives = {
            (int(record.src_index), int(record.dst_index))
            for record in records
            if int(record.label) == 0
        }
        positives = {
            (int(record.src_index), int(record.dst_index))
            for record in records
            if int(record.label) == 1
        }
        assert 0 in labels
        assert 1 in labels
        assert positives.isdisjoint(negatives)
        assert negatives.isdisjoint(positive_edges)


def test_random_link_split_records_expose_stable_sample_ids_and_grouped_query_identifiers():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        seed=9,
    )(graph)

    split_datasets = {
        "train": list(train_dataset),
        "val": list(val_dataset),
        "test": list(test_dataset),
    }
    records = [record for dataset in split_datasets.values() for record in dataset]
    positive_records = [record for record in records if int(record.label) == 1]

    assert all(record.sample_id is not None for record in records)
    assert all(record.metadata["sample_id"] == record.sample_id for record in records)
    assert all(record.metadata["query_id"] == record.query_id for record in records)
    assert all(record.query_id == record.sample_id for record in positive_records)

    for split_name, dataset in split_datasets.items():
        positive_query_ids = {
            record.query_id
            for record in dataset
            if int(record.label) == 1
        }
        negative_records = [record for record in dataset if int(record.label) == 0]
        assert all(record.query_id in positive_query_ids for record in negative_records)
        assert all(record.query_id != record.sample_id for record in negative_records)


def test_random_link_split_validation_negatives_stay_query_local_for_ranking_style_batches():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )

    _, val_dataset, _ = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=False,
        seed=0,
    )(graph)

    positive_records = [record for record in val_dataset if int(record.label) == 1]
    negative_records = [record for record in val_dataset if int(record.label) == 0]

    assert len(positive_records) == 1
    assert len(negative_records) == 2
    assert all(int(record.src_index) == int(positive_records[0].src_index) for record in negative_records)
    assert all(record.query_id == positive_records[0].query_id for record in negative_records)


def test_random_link_split_accepts_tensor_negative_sampling_ratio_without_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )

    def fail_int(self):
        raise AssertionError("RandomLinkSplit neg_sampling_ratio should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    _, val_dataset, _ = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=torch.tensor(2),
        add_negative_train_samples=False,
        seed=0,
    )(graph)

    positive_records = [record for record in val_dataset if int(record.label) == 1]
    negative_records = [record for record in val_dataset if int(record.label) == 0]

    assert len(positive_records) == 1
    assert len(negative_records) == 2
    assert all(record.query_id == positive_records[0].query_id for record in negative_records)


def test_random_link_split_validation_negatives_avoid_duplicate_destinations_when_unique_candidates_exist():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 0, 1, 1, 2, 3], [1, 2, 3, 2, 3, 3, 0]]),
        x=torch.randn(5, 3),
    )

    _, val_dataset, _ = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=4.0,
        add_negative_train_samples=False,
        seed=0,
    )(graph)

    positive_records = [record for record in val_dataset if int(record.label) == 1]
    negative_records = [record for record in val_dataset if int(record.label) == 0]
    negative_dst = [int(record.dst_index) for record in negative_records]

    assert len(positive_records) == 1
    assert len(negative_records) == 4
    assert len(set(negative_dst)) == len(negative_dst)


def test_random_link_split_train_negatives_stay_query_local_when_added_to_split():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 3], [1, 2, 2, 3, 3, 0]]),
        x=torch.randn(4, 3),
    )

    train_dataset, _, _ = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        seed=0,
    )(graph)

    positive_records = [record for record in train_dataset if int(record.label) == 1]
    negative_records = [record for record in train_dataset if int(record.label) == 0]
    positive_by_query = {record.query_id: record for record in positive_records}

    assert len(positive_records) == len(negative_records)
    assert all(record.query_id in positive_by_query for record in negative_records)
    assert all(record.query_id != record.sample_id for record in negative_records)
    assert all(int(record.src_index) == int(positive_by_query[record.query_id].src_index) for record in negative_records)


def test_random_link_split_marks_positive_records_to_exclude_seed_edges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]]),
        x=torch.randn(4, 3),
    )

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        seed=9,
    )(graph)

    records = list(train_dataset) + list(val_dataset) + list(test_dataset)
    positive_records = [record for record in records if int(record.label) == 1]
    negative_records = [record for record in records if int(record.label) == 0]

    assert all(record.exclude_seed_edge for record in positive_records)
    assert all(record.metadata["exclude_seed_edges"] is True for record in positive_records)
    assert all(not record.exclude_seed_edge for record in negative_records)
    assert all("exclude_seed_edges" not in record.metadata for record in negative_records)


def test_random_link_split_avoids_tensor_tolist_for_grouping_and_negative_sampling(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        x=torch.randn(4, 3),
    )

    def fail_tolist(self):
        raise AssertionError("RandomLinkSplit should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        is_undirected=True,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        seed=7,
    )(graph)

    assert len(list(train_dataset)) > 0
    assert len(list(val_dataset)) > 0
    assert len(list(test_dataset)) > 0


def test_random_link_split_avoids_tensor_item_for_grouping_and_negative_sampling(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        x=torch.randn(4, 3),
    )

    def fail_item(self):
        raise AssertionError("RandomLinkSplit should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        is_undirected=True,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        seed=7,
    )(graph)

    assert len(list(train_dataset)) > 0
    assert len(list(val_dataset)) > 0
    assert len(list(test_dataset)) > 0


def test_random_link_split_avoids_tensor_int_for_grouping(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        x=torch.randn(4, 3),
    )

    def fail_int(self):
        raise AssertionError("RandomLinkSplit grouping should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        is_undirected=True,
        seed=7,
    )(graph)

    assert len(list(train_dataset)) > 0
    assert len(list(val_dataset)) > 0
    assert len(list(test_dataset)) > 0


def test_random_link_split_accepts_tensor_seed_without_tensor_int(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        x=torch.randn(4, 3),
    )

    def fail_int(self):
        raise AssertionError("RandomLinkSplit should stay off tensor.__int__ for tensor seed values")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        is_undirected=True,
        seed=torch.tensor(7),
    )(graph)

    assert len(list(train_dataset)) > 0
    assert len(list(val_dataset)) > 0
    assert len(list(test_dataset)) > 0


def test_random_link_split_avoids_torch_unique_for_grouping_and_negative_sampling(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]]),
        x=torch.randn(4, 3),
    )

    def fail_unique(*args, **kwargs):
        raise AssertionError("RandomLinkSplit should avoid torch.unique")

    monkeypatch.setattr(torch, "unique", fail_unique)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        is_undirected=True,
        neg_sampling_ratio=1.0,
        add_negative_train_samples=True,
        seed=7,
    )(graph)

    assert len(list(train_dataset)) > 0
    assert len(list(val_dataset)) > 0
    assert len(list(test_dataset)) > 0


def test_random_link_split_avoids_dense_full_for_validation_negative_queries(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 0, 1, 1, 2, 3], [1, 2, 3, 2, 3, 3, 0]]),
        x=torch.randn(5, 3),
    )
    original_full = torch.full

    def guarded_full(*args, **kwargs):
        caller = inspect.currentframe().f_back
        size = args[0] if args else kwargs.get("size")
        if size == (graph.x.size(0),) and caller is not None and caller.f_code.co_name == "_sample_negative_destinations_for_source":
            raise AssertionError("RandomLinkSplit negative destination sampling should avoid dense torch.full queries")
        return original_full(*args, **kwargs)

    monkeypatch.setattr(torch, "full", guarded_full)

    _, val_dataset, _ = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=4.0,
        add_negative_train_samples=False,
        seed=0,
    )(graph)

    negative_records = [record for record in val_dataset if int(record.label) == 0]

    assert len(negative_records) == 4


def test_random_link_split_avoids_tensor_int_for_negative_sampling(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 0, 1, 1, 2, 3], [1, 2, 3, 2, 3, 3, 0]]),
        x=torch.randn(5, 3),
    )

    def fail_int(self):
        raise AssertionError("RandomLinkSplit negative sampling should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    _, val_dataset, _ = RandomLinkSplit(
        num_val=1,
        num_test=1,
        neg_sampling_ratio=4.0,
        add_negative_train_samples=False,
        seed=0,
    )(graph)

    negative_records = [record for record in val_dataset if int(record.label) == 0]

    assert len(negative_records) == 4


def test_random_link_split_attach_negative_records_avoids_tensor_int_for_tensor_seed_indices(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
        x=torch.randn(4, 3),
    )
    edge_type = next(iter(graph.edges))
    splitter = RandomLinkSplit(num_val=1, num_test=1, neg_sampling_ratio=1.0)
    records = [
        LinkPredictionRecord(
            graph=graph,
            src_index=torch.tensor(0),
            dst_index=torch.tensor(1),
            label=1,
            metadata={"split": "val"},
            sample_id="val:0",
            query_id="q0",
            edge_type=edge_type,
        )
    ]

    def fail_int(self):
        raise AssertionError("RandomLinkSplit negative record attachment should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    augmented = splitter._attach_negative_records(
        graph=graph,
        split="val",
        records=records,
        edge_type=edge_type,
        reverse_edge_type=None,
        num_src_nodes=graph.x.size(0),
        num_dst_nodes=graph.x.size(0),
        excluded_edges=torch.tensor([1, 2], dtype=torch.long),
        generator=torch.Generator().manual_seed(0),
    )

    assert len(augmented) == 2
    negative_record = augmented[1]
    assert negative_record.src_index == 0
    assert negative_record.label == 0
    assert negative_record.metadata["negative_sampled"] is True
    assert negative_record.query_id == "q0"


def test_random_link_split_negative_destination_sampling_avoids_tensor_int_for_tensor_seed_indices(monkeypatch):
    splitter = RandomLinkSplit(num_val=1, num_test=1, neg_sampling_ratio=1.0)

    def fail_int(self):
        raise AssertionError("RandomLinkSplit negative destination sampling should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    destinations = splitter._sample_negative_destinations_for_source(
        src_index=torch.tensor(0),
        count=2,
        num_dst_nodes=4,
        excluded_edges=torch.tensor([1, 2], dtype=torch.long),
        generator=torch.Generator().manual_seed(0),
    )

    assert len(destinations) == 2
    assert set(destinations) == {0, 3}


def test_random_link_split_negative_edge_sampling_avoids_tensor_int_for_sampled_edge_keys(monkeypatch):
    splitter = RandomLinkSplit(num_val=1, num_test=1, neg_sampling_ratio=1.0)

    def fail_int(self):
        raise AssertionError("RandomLinkSplit negative edge sampling should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    sampled = splitter._sample_negative_edges(
        count=2,
        num_src_nodes=2,
        num_dst_nodes=3,
        excluded_edges=torch.empty(0, dtype=torch.long),
        generator=torch.Generator().manual_seed(0),
    )

    assert len(sampled) == 2
    assert all(isinstance(edge, tuple) and len(edge) == 2 for edge in sampled)


def test_random_link_split_avoids_tensor_tolist_for_reverse_edge_alignment(monkeypatch):
    edge_type = ("author", "writes", "paper")
    reverse_edge_type = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1, 2, 1], [1, 2, 3, 0]])},
            reverse_edge_type: {"edge_index": torch.tensor([[1, 2, 3, 0], [0, 1, 2, 1]])},
        },
    )

    def fail_tolist(self):
        raise AssertionError("RandomLinkSplit reverse-edge alignment should stay off tensor.tolist")

    monkeypatch.setattr(torch.Tensor, "tolist", fail_tolist)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        edge_type=edge_type,
        rev_edge_type=reverse_edge_type,
        seed=13,
    )(graph)

    assert len(list(train_dataset)) > 0
    assert len(list(val_dataset)) > 0
    assert len(list(test_dataset)) > 0


def test_random_link_split_avoids_tensor_item_for_reverse_edge_alignment(monkeypatch):
    edge_type = ("author", "writes", "paper")
    reverse_edge_type = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1, 2, 1], [1, 2, 3, 0]])},
            reverse_edge_type: {"edge_index": torch.tensor([[1, 2, 3, 0], [0, 1, 2, 1]])},
        },
    )

    def fail_item(self):
        raise AssertionError("RandomLinkSplit reverse-edge alignment should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        edge_type=edge_type,
        rev_edge_type=reverse_edge_type,
        seed=13,
    )(graph)

    assert len(list(train_dataset)) > 0
    assert len(list(val_dataset)) > 0
    assert len(list(test_dataset)) > 0


def test_random_link_split_avoids_tensor_int_for_reverse_edge_alignment(monkeypatch):
    edge_type = ("author", "writes", "paper")
    reverse_edge_type = ("paper", "written_by", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.randn(3, 4)},
            "paper": {"x": torch.randn(4, 4)},
        },
        edges={
            edge_type: {"edge_index": torch.tensor([[0, 1, 2, 1], [1, 2, 3, 0]])},
            reverse_edge_type: {"edge_index": torch.tensor([[1, 2, 3, 0], [0, 1, 2, 1]])},
        },
    )

    def fail_int(self):
        raise AssertionError("RandomLinkSplit reverse-edge alignment should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    train_dataset, val_dataset, test_dataset = RandomLinkSplit(
        num_val=1,
        num_test=1,
        edge_type=edge_type,
        rev_edge_type=reverse_edge_type,
        seed=13,
    )(graph)

    assert len(list(train_dataset)) > 0
    assert len(list(val_dataset)) > 0
    assert len(list(test_dataset)) > 0
