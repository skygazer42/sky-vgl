import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, resolve_node_feature


class PointNetConv(nn.Module):
    def __init__(self, in_channels, out_channels, pos_channels, aggr="max"):
        super().__init__()
        if aggr not in {"max", "sum", "mean"}:
            raise ValueError("PointNetConv only supports max, sum, and mean aggregation")

        self.out_channels = out_channels
        self.aggr = aggr
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channels + pos_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels),
        )
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None, pos=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "PointNetConv")
        pos = resolve_node_feature(graph_or_x, use_graph_data, pos, "pos", "PointNetConv")
        row, col = edge_index
        relative = pos[row] - pos[col]
        messages = self.local_mlp(torch.cat([x[row], relative], dim=-1))

        if self.aggr == "max":
            out = torch.full((x.size(0), self.out_channels), float("-inf"), dtype=x.dtype, device=x.device)
            index = col.unsqueeze(-1).expand(-1, self.out_channels)
            out.scatter_reduce_(0, index, messages, reduce="amax", include_self=True)
            out = torch.where(torch.isneginf(out), torch.zeros_like(out), out)
        else:
            out = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
            out.index_add_(0, col, messages)
            if self.aggr == "mean":
                degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
                degree.index_add_(0, col, torch.ones(col.size(0), dtype=x.dtype, device=x.device))
                out = out / degree.clamp_min(1).unsqueeze(-1)

        return out + self.root_linear(x)
