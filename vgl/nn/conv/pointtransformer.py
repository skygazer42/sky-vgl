import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax, resolve_node_feature


class PointTransformerConv(nn.Module):
    def __init__(self, in_channels, out_channels, pos_channels):
        super().__init__()
        self.out_channels = out_channels
        self.query_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.key_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.value_linear = nn.Linear(in_channels, out_channels, bias=False)
        self.pos_mlp = nn.Sequential(
            nn.Linear(pos_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.attn_linear = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1),
        )
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None, pos=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "PointTransformerConv")
        pos = resolve_node_feature(graph_or_x, use_graph_data, pos, "pos", "PointTransformerConv")
        row, col = edge_index

        query = self.query_linear(x)
        key = self.key_linear(x)
        value = self.value_linear(x)
        delta = self.pos_mlp(pos[row] - pos[col])

        logits = self.attn_linear(query[col] - key[row] + delta).squeeze(-1)
        weights = edge_softmax(logits, edge_index, x.size(0))
        messages = (value[row] + delta) * weights.unsqueeze(-1)

        out = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
        out.index_add_(0, col, messages)
        return out + self.root_linear(x)
