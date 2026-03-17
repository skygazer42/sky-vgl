import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs
from vgl.nn.conv.nnconv import _resolve_edge_features


def _append_self_loops(edge_index, edge_attr, num_nodes):
    loop_index = torch.arange(num_nodes, device=edge_index.device)
    loop_edge_index = torch.stack([loop_index, loop_index], dim=0)
    loop_edge_attr = torch.zeros(num_nodes, edge_attr.size(-1), dtype=edge_attr.dtype, device=edge_attr.device)
    return (
        torch.cat([edge_index, loop_edge_index], dim=1),
        torch.cat([edge_attr, loop_edge_attr], dim=0),
    )


class PDNConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        edge_channels,
        hidden_channels=None,
        add_self_loops=True,
        normalize=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.add_self_loops = add_self_loops
        self.normalize = normalize

        hidden_channels = hidden_channels or max(edge_channels, out_channels)
        self.node_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid(),
        )
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None, edge_attr=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "PDNConv")
        edge_attr = _resolve_edge_features(graph_or_x, use_graph_data, edge_attr, "edge_attr", "PDNConv")
        if self.add_self_loops:
            edge_index, edge_attr = _append_self_loops(edge_index, edge_attr, x.size(0))

        row, col = edge_index
        messages = self.node_linear(x[row]) * self.edge_mlp(edge_attr)
        out = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
        out.index_add_(0, col, messages)

        if self.normalize:
            degree = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
            degree.index_add_(0, col, torch.ones_like(messages))
            out = out / degree.clamp_min(1.0)

        return out + self.root_linear(x)
