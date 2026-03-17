import torch

from vgl import Graph


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
