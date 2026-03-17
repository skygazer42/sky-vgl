import pytest
import torch

from vgl import Graph
from vgl.nn import GroupRevRes
from vgl.nn.conv.lg import LGConv


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


def test_grouprevres_accepts_graph_input():
    block = GroupRevRes(LGConv(), num_groups=2)

    out = block(_homo_graph())

    assert out.shape == (3, 4)


def test_grouprevres_accepts_x_and_edge_index():
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    block = GroupRevRes(torch.nn.ModuleList([LGConv(), LGConv()]))

    out = block(x, edge_index)

    assert out.shape == (3, 4)


def test_grouprevres_inverse_reconstructs_input():
    torch.manual_seed(19)
    x = torch.randn(3, 4)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]])
    block = GroupRevRes(LGConv(normalize=False), num_groups=2)

    out = block(x, edge_index)
    recovered = block.inverse(out, edge_index)

    assert torch.allclose(recovered, x, atol=1e-6, rtol=1e-5)


def test_grouprevres_rejects_invalid_num_groups():
    with pytest.raises(ValueError, match="groups|2"):
        GroupRevRes(LGConv(), num_groups=1)


def test_grouprevres_rejects_non_divisible_channels():
    x = torch.randn(3, 5)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
    block = GroupRevRes(LGConv(), num_groups=2)

    with pytest.raises(ValueError, match="divisible"):
        block(x, edge_index)


def test_grouprevres_rejects_unsupported_wrapped_forward_contract():
    class NeedsExtraArg(torch.nn.Module):
        def forward(self, x, edge_index, extra):
            return x

    with pytest.raises(ValueError, match="forward|extra|runtime"):
        GroupRevRes(NeedsExtraArg(), num_groups=2)


def test_grouprevres_rejects_hetero_graph_input():
    block = GroupRevRes(LGConv(), num_groups=2)

    with pytest.raises(ValueError, match="homogeneous"):
        block(_hetero_graph())
