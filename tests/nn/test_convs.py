import pytest
import torch

from vgl import Graph
from vgl.nn.conv.cheb import ChebConv
from vgl.nn.conv.appnp import APPNPConv
from vgl.nn.conv.gcn import GCNConv
from vgl.nn.conv.gatv2 import GATv2Conv
from vgl.nn.conv.gin import GINConv
from vgl.nn.conv.sg import SGConv
from vgl.nn.conv.sage import SAGEConv
from vgl.nn.conv.tag import TAGConv


def _homo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
    )


def _hetero_graph():
    return Graph.hetero(
        nodes={"paper": {"x": torch.randn(2, 4)}, "author": {"x": torch.randn(2, 4)}},
        edges={("author", "writes", "paper"): {"edge_index": torch.tensor([[0, 1], [1, 0]])}},
    )


def test_gcn_conv_accepts_graph_input():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        x=torch.randn(2, 4),
        y=torch.tensor([0, 1]),
    )
    conv = GCNConv(in_channels=4, out_channels=3)

    out = conv(graph)

    assert out.shape == (2, 3)


def test_sage_conv_accepts_x_and_edge_index():
    x = torch.randn(2, 4)
    edge_index = torch.tensor([[0, 1], [1, 0]])
    conv = SAGEConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (2, 3)


def test_gin_conv_accepts_graph_input():
    conv = GINConv(in_channels=4, out_channels=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_gin_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = GINConv(in_channels=4, out_channels=3)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_gatv2_conv_respects_head_output_shapes():
    graph = _homo_graph()

    concat_conv = GATv2Conv(in_channels=4, out_channels=3, heads=2, concat=True)
    mean_conv = GATv2Conv(in_channels=4, out_channels=3, heads=2, concat=False)

    concat_out = concat_conv(graph)
    mean_out = mean_conv(graph)

    assert concat_out.shape == (3, 6)
    assert mean_out.shape == (3, 3)


def test_appnp_conv_accepts_graph_input():
    conv = APPNPConv(in_channels=4, out_channels=3, steps=2, alpha=0.1)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_tag_conv_accepts_graph_input():
    conv = TAGConv(in_channels=4, out_channels=3, k=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_tag_conv_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    conv = TAGConv(in_channels=4, out_channels=3, k=2)

    out = conv(x, edge_index)

    assert out.shape == (3, 3)


def test_sg_conv_accepts_graph_input():
    conv = SGConv(in_channels=4, out_channels=3, k=2)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


def test_cheb_conv_accepts_graph_input():
    conv = ChebConv(in_channels=4, out_channels=3, k=3)

    out = conv(_homo_graph())

    assert out.shape == (3, 3)


@pytest.mark.parametrize(
    ("conv_cls", "kwargs"),
    [
        (GINConv, {"in_channels": 4, "out_channels": 3}),
        (GATv2Conv, {"in_channels": 4, "out_channels": 3}),
        (APPNPConv, {"in_channels": 4, "out_channels": 3}),
        (TAGConv, {"in_channels": 4, "out_channels": 3}),
        (SGConv, {"in_channels": 4, "out_channels": 3}),
        (ChebConv, {"in_channels": 4, "out_channels": 3}),
    ],
)
def test_new_homo_convs_reject_hetero_graph_input(conv_cls, kwargs):
    conv = conv_cls(**kwargs)

    with pytest.raises(ValueError, match="homogeneous"):
        conv(_hetero_graph())

