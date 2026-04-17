import torch
import pytest

from vgl import Graph
from vgl.graph.errors import GraphConstructionError
from vgl.graph.graph import GRAPH_FORMAT, GRAPH_FORMAT_VERSION


def test_homo_graph_exposes_pyg_style_fields():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    x = torch.randn(2, 4)
    y = torch.tensor([0, 1])

    graph = Graph.homo(edge_index=edge_index, x=x, y=y)

    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.x, x)
    assert torch.equal(graph.y, y)
    assert torch.equal(graph.ndata["x"], x)


def test_homo_graph_accepts_edge_data():
    edge_index = torch.tensor([[0, 1], [1, 0]])
    edge_attr = torch.randn(2, 3)
    edge_weight = torch.tensor([0.5, 1.5])

    graph = Graph.homo(
        edge_index=edge_index,
        x=torch.randn(2, 4),
        edge_data={"edge_attr": edge_attr, "edge_weight": edge_weight},
    )

    assert torch.equal(graph.edge_index, edge_index)
    assert torch.equal(graph.edata["edge_attr"], edge_attr)
    assert torch.equal(graph.edata["edge_weight"], edge_weight)
    assert graph.schema.edge_features[("node", "to", "node")] == (
        "edge_index",
        "edge_attr",
        "edge_weight",
    )


def test_graph_artifact_metadata_uses_shared_format_keys():
    graph = Graph.homo(edge_index=torch.tensor([[0, 1], [1, 0]]), x=torch.randn(2, 4))

    metadata = graph.artifact_metadata()

    assert metadata["format"] == GRAPH_FORMAT
    assert metadata["format_version"] == GRAPH_FORMAT_VERSION
    assert metadata["node_types"] == ("node",)
    assert metadata["edge_types"] == (("node", "to", "node"),)
    assert metadata["time_attr"] is None


def test_homo_graph_rejects_non_2_by_num_edges_edge_index():
    with pytest.raises(GraphConstructionError, match="edge_index must have shape \\(2, num_edges\\)"):
        Graph.homo(edge_index=torch.tensor([0, 1, 2]), x=torch.randn(3, 4))


def test_homo_graph_rejects_node_feature_length_mismatch():
    with pytest.raises(GraphConstructionError, match="node feature 'y' must match node count 2"):
        Graph.homo(
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            x=torch.randn(2, 4),
            y=torch.tensor([0, 1, 2]),
        )
