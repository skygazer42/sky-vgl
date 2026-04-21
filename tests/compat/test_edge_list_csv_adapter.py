import csv

import pytest
import torch

from vgl import Graph


def test_graph_round_trips_to_edge_list_csv(tmp_path):
    path = tmp_path / "graph.csv"
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 2]]),
        edge_data={
            "edge_weight": torch.tensor([0.5, 1.5, 2.5]),
            "e_id": torch.tensor([10, 11, 12]),
        },
    )

    graph.to_edge_list_csv(path)
    restored = Graph.from_edge_list_csv(path)

    rows = list(csv.DictReader(path.read_text().splitlines()))

    assert rows == [
        {"src": "0", "dst": "1", "edge_weight": "0.5", "e_id": "10"},
        {"src": "0", "dst": "2", "edge_weight": "1.5", "e_id": "11"},
        {"src": "2", "dst": "2", "edge_weight": "2.5", "e_id": "12"},
    ]
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert torch.equal(restored.edata["edge_weight"], graph.edata["edge_weight"])
    assert torch.equal(restored.edata["e_id"], graph.edata["e_id"])


def test_to_edge_list_csv_avoids_tensor_item(monkeypatch, tmp_path):
    path = tmp_path / "graph.csv"
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 2]]),
        edge_data={
            "edge_weight": torch.tensor([0.5, 1.5, 2.5]),
            "e_id": torch.tensor([10, 11, 12]),
        },
    )

    def fail_item(self):
        raise AssertionError("edge list CSV export should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    graph.to_edge_list_csv(path)

    rows = list(csv.DictReader(path.read_text().splitlines()))

    assert rows[0] == {"src": "0", "dst": "1", "edge_weight": "0.5", "e_id": "10"}


def test_to_edge_list_csv_avoids_tensor_int(monkeypatch, tmp_path):
    path = tmp_path / "graph.csv"
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 2]]),
        edge_data={
            "edge_weight": torch.tensor([0.5, 1.5, 2.5]),
            "e_id": torch.tensor([10, 11, 12]),
        },
    )

    def fail_int(self):
        raise AssertionError("edge list CSV export should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    graph.to_edge_list_csv(path)

    rows = list(csv.DictReader(path.read_text().splitlines()))

    assert rows[0] == {"src": "0", "dst": "1", "edge_weight": "0.5", "e_id": "10"}


def test_edge_list_csv_supports_custom_columns_and_delimiter(tmp_path):
    path = tmp_path / "graph.tsv"
    path.write_text("source|target|weight\n0|1|0.5\n1|2|1.5\n")

    graph = Graph.from_edge_list_csv(path, src_column="source", dst_column="target", delimiter="|")
    exported = tmp_path / "export.tsv"
    graph.to_edge_list_csv(
        exported,
        src_column="u",
        dst_column="v",
        edge_columns=["weight"],
        delimiter="|",
    )

    assert torch.equal(graph.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(graph.edata["weight"], torch.tensor([0.5, 1.5]))
    assert exported.read_text() == "u|v|weight\n0|1|0.5\n1|2|1.5\n"


def test_graph_from_edge_list_csv_preserves_explicit_num_nodes_for_isolates(tmp_path):
    path = tmp_path / "isolates.csv"
    path.write_text("src,dst\n0,1\n")

    graph = Graph.from_edge_list_csv(path, num_nodes=4)

    assert graph.num_nodes() == 4
    assert torch.equal(graph.n_id, torch.tensor([0, 1, 2, 3]))
    assert graph.adjacency().shape == (4, 4)
    assert torch.equal(graph.to_edge_list(), torch.tensor([[0, 1]]))


def test_graph_from_edge_list_csv_accepts_header_only_file(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("src,dst\n")

    graph = Graph.from_edge_list_csv(path)

    assert graph.num_nodes() == 0
    assert torch.equal(graph.edge_index, torch.empty((2, 0), dtype=torch.long))
    assert torch.equal(graph.to_edge_list(), torch.empty((0, 2), dtype=torch.long))


def test_from_edge_list_csv_rejects_non_numeric_edge_features(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_text("src,dst,label\n0,1,blue\n1,2,green\n")

    with pytest.raises(ValueError, match="numeric"):
        Graph.from_edge_list_csv(path)


def test_to_edge_list_csv_rejects_heterogeneous_graphs(tmp_path):
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[3.0], [4.0]])},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [0, 1]]),
            }
        },
    )

    with pytest.raises(ValueError, match="homogeneous"):
        graph.to_edge_list_csv(tmp_path / "hetero.csv")


def test_to_edge_list_csv_rejects_duplicate_public_edge_ids(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0], [1, 1]]),
        edge_data={"e_id": torch.tensor([7, 7])},
    )

    with pytest.raises(ValueError, match="unique public ids"):
        graph.to_edge_list_csv(tmp_path / "graph.csv")


def test_compat_exports_edge_list_csv_helpers(tmp_path):
    from vgl.compat import from_edge_list_csv, to_edge_list_csv

    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        edge_data={"edge_weight": torch.tensor([0.5])},
    )
    path = tmp_path / "graph.csv"

    to_edge_list_csv(graph, path)
    restored = from_edge_list_csv(path)

    assert path.read_text() == "src,dst,edge_weight\n0,1,0.5\n"
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert torch.equal(restored.edata["edge_weight"], graph.edata["edge_weight"])
