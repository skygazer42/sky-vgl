import csv

import pytest
import torch

from vgl import Graph


def test_graph_round_trips_to_csv_tables(tmp_path):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 1]]),
        n_id=torch.tensor([10, 20, 30]),
        x=torch.tensor([1.0, 2.0, 3.0]),
        y=torch.tensor([0, 1, 0]),
        edge_data={
            "edge_weight": torch.tensor([0.5, 1.5, 2.5]),
            "e_id": torch.tensor([11, 12, 13]),
        },
    )

    graph.to_csv_tables(nodes_path, edges_path)
    restored = Graph.from_csv_tables(nodes_path, edges_path)

    node_rows = list(csv.DictReader(nodes_path.read_text().splitlines()))
    edge_rows = list(csv.DictReader(edges_path.read_text().splitlines()))

    assert node_rows == [
        {"node_id": "10", "x": "1.0", "y": "0"},
        {"node_id": "20", "x": "2.0", "y": "1"},
        {"node_id": "30", "x": "3.0", "y": "0"},
    ]
    assert edge_rows == [
        {"src": "10", "dst": "20", "edge_weight": "0.5", "e_id": "11"},
        {"src": "10", "dst": "30", "edge_weight": "1.5", "e_id": "12"},
        {"src": "30", "dst": "20", "edge_weight": "2.5", "e_id": "13"},
    ]
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert torch.equal(restored.n_id, graph.n_id)
    assert torch.equal(restored.x, graph.x)
    assert torch.equal(restored.y, graph.y)
    assert torch.equal(restored.edata["edge_weight"], graph.edata["edge_weight"])
    assert torch.equal(restored.edata["e_id"], graph.edata["e_id"])


def test_to_csv_tables_avoids_tensor_item(monkeypatch, tmp_path):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 1]]),
        n_id=torch.tensor([10, 20, 30]),
        x=torch.tensor([1.0, 2.0, 3.0]),
        y=torch.tensor([0, 1, 0]),
        edge_data={
            "edge_weight": torch.tensor([0.5, 1.5, 2.5]),
            "e_id": torch.tensor([11, 12, 13]),
        },
    )

    def fail_item(self):
        raise AssertionError("CSV table export should stay off tensor.item")

    monkeypatch.setattr(torch.Tensor, "item", fail_item)

    graph.to_csv_tables(nodes_path, edges_path)

    node_rows = list(csv.DictReader(nodes_path.read_text().splitlines()))
    edge_rows = list(csv.DictReader(edges_path.read_text().splitlines()))

    assert node_rows[0] == {"node_id": "10", "x": "1.0", "y": "0"}
    assert edge_rows[0] == {"src": "10", "dst": "20", "edge_weight": "0.5", "e_id": "11"}


def test_to_csv_tables_avoids_tensor_int(monkeypatch, tmp_path):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 2], [1, 2, 1]]),
        n_id=torch.tensor([10, 20, 30]),
        x=torch.tensor([1.0, 2.0, 3.0]),
        y=torch.tensor([0, 1, 0]),
        edge_data={
            "edge_weight": torch.tensor([0.5, 1.5, 2.5]),
            "e_id": torch.tensor([11, 12, 13]),
        },
    )

    def fail_int(self):
        raise AssertionError("CSV table export should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    graph.to_csv_tables(nodes_path, edges_path)

    node_rows = list(csv.DictReader(nodes_path.read_text().splitlines()))
    edge_rows = list(csv.DictReader(edges_path.read_text().splitlines()))

    assert node_rows[0] == {"node_id": "10", "x": "1.0", "y": "0"}
    assert edge_rows[0] == {"src": "10", "dst": "20", "edge_weight": "0.5", "e_id": "11"}


def test_csv_tables_support_custom_columns_and_delimiter(tmp_path):
    nodes_path = tmp_path / "nodes.tsv"
    edges_path = tmp_path / "edges.tsv"
    nodes_path.write_text("node|score\n10|1.5\n20|2.5\n30|3.5\n")
    edges_path.write_text("source|target|weight\n10|20|0.5\n30|20|1.5\n")

    graph = Graph.from_csv_tables(
        nodes_path,
        edges_path,
        node_id_column="node",
        src_column="source",
        dst_column="target",
        delimiter="|",
    )
    exported_nodes = tmp_path / "export_nodes.tsv"
    exported_edges = tmp_path / "export_edges.tsv"
    graph.to_csv_tables(
        exported_nodes,
        exported_edges,
        node_id_column="id",
        src_column="u",
        dst_column="v",
        node_columns=["score"],
        edge_columns=["weight"],
        delimiter="|",
    )

    assert torch.equal(graph.n_id, torch.tensor([10, 20, 30]))
    assert torch.equal(graph.edge_index, torch.tensor([[0, 2], [1, 1]]))
    assert torch.equal(graph.x if hasattr(graph, "x") else torch.tensor([]), torch.tensor([]))
    assert torch.equal(graph.ndata["score"], torch.tensor([1.5, 2.5, 3.5]))
    assert torch.equal(graph.edata["weight"], torch.tensor([0.5, 1.5]))
    assert exported_nodes.read_text() == "id|score\n10|1.5\n20|2.5\n30|3.5\n"
    assert exported_edges.read_text() == "u|v|weight\n10|20|0.5\n30|20|1.5\n"


def test_graph_from_csv_tables_preserves_isolates_from_node_rows(tmp_path):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    nodes_path.write_text("node_id,x\n10,1.0\n20,2.0\n30,3.0\n")
    edges_path.write_text("src,dst\n")

    graph = Graph.from_csv_tables(nodes_path, edges_path)

    assert graph.num_nodes() == 3
    assert torch.equal(graph.n_id, torch.tensor([10, 20, 30]))
    assert torch.equal(graph.x, torch.tensor([1.0, 2.0, 3.0]))
    assert graph.adjacency().shape == (3, 3)
    assert torch.equal(graph.edge_index, torch.empty((2, 0), dtype=torch.long))


def test_from_csv_tables_rejects_unknown_edge_node_ids(tmp_path):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    nodes_path.write_text("node_id\n10\n20\n")
    edges_path.write_text("src,dst\n10,30\n")

    with pytest.raises(ValueError, match="unknown node id"):
        Graph.from_csv_tables(nodes_path, edges_path)


def test_from_csv_tables_rejects_non_numeric_node_features(tmp_path):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    nodes_path.write_text("node_id,label\n10,blue\n20,green\n")
    edges_path.write_text("src,dst\n10,20\n")

    with pytest.raises(ValueError, match="numeric"):
        Graph.from_csv_tables(nodes_path, edges_path)


def test_from_csv_tables_rejects_non_numeric_edge_features(tmp_path):
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"
    nodes_path.write_text("node_id\n10\n20\n")
    edges_path.write_text("src,dst,label\n10,20,blue\n")

    with pytest.raises(ValueError, match="numeric"):
        Graph.from_csv_tables(nodes_path, edges_path)


def test_to_csv_tables_rejects_heterogeneous_graphs(tmp_path):
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
        graph.to_csv_tables(tmp_path / "nodes.csv", tmp_path / "edges.csv")


def test_to_csv_tables_rejects_duplicate_public_node_ids(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([1.0, 2.0]),
        n_id=torch.tensor([10, 10]),
    )

    with pytest.raises(ValueError, match="unique public node ids"):
        graph.to_csv_tables(tmp_path / "nodes.csv", tmp_path / "edges.csv")


def test_to_csv_tables_rejects_non_integral_public_node_ids(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        x=torch.tensor([1.0, 2.0]),
        n_id=torch.tensor([1.5, 2.5]),
    )

    with pytest.raises(ValueError, match="integer public node ids"):
        graph.to_csv_tables(tmp_path / "nodes.csv", tmp_path / "edges.csv")


def test_to_csv_tables_rejects_duplicate_public_edge_ids(tmp_path):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0], [1, 1]]),
        x=torch.tensor([1.0, 2.0]),
        n_id=torch.tensor([10, 20]),
        edge_data={"e_id": torch.tensor([7, 7])},
    )

    with pytest.raises(ValueError, match="unique public ids"):
        graph.to_csv_tables(tmp_path / "nodes.csv", tmp_path / "edges.csv")


def test_compat_exports_csv_table_helpers(tmp_path):
    from vgl.compat import from_csv_tables, to_csv_tables

    graph = Graph.homo(
        edge_index=torch.tensor([[0], [1]]),
        n_id=torch.tensor([10, 20]),
        x=torch.tensor([1.0, 2.0]),
        edge_data={"edge_weight": torch.tensor([0.5])},
    )
    nodes_path = tmp_path / "nodes.csv"
    edges_path = tmp_path / "edges.csv"

    to_csv_tables(graph, nodes_path, edges_path)
    restored = from_csv_tables(nodes_path, edges_path)

    assert nodes_path.read_text() == "node_id,x\n10,1.0\n20,2.0\n"
    assert edges_path.read_text() == "src,dst,edge_weight\n10,20,0.5\n"
    assert torch.equal(restored.n_id, graph.n_id)
    assert torch.equal(restored.edge_index, graph.edge_index)
    assert torch.equal(restored.x, graph.x)
    assert torch.equal(restored.edata["edge_weight"], graph.edata["edge_weight"])
