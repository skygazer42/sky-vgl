import torch
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs


def _resolve_edge_features(graph_or_x, use_graph_data, edge_attr, key, layer_name):
    if edge_attr is not None:
        return edge_attr
    if use_graph_data:
        if key in graph_or_x.edata:
            return graph_or_x.edata[key]
        raise ValueError(f"{layer_name} requires {key} in graph.edata or as an explicit argument")
    raise ValueError(f"{layer_name} requires {key} when called with x and edge_index")


class NNConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels, hidden_channels=None, aggr="mean"):
        super().__init__()
        if aggr not in {"sum", "mean"}:
            raise ValueError("NNConv only supports sum and mean aggregation")

        self.out_channels = out_channels
        self.aggr = aggr
        hidden = hidden_channels or max(edge_channels, in_channels * out_channels)
        self.edge_network = nn.Sequential(
            nn.Linear(edge_channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_channels * out_channels),
        )
        self.root_linear = nn.Linear(in_channels, out_channels)

    def forward(self, graph_or_x, edge_index=None, edge_attr=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "NNConv")
        edge_attr = _resolve_edge_features(graph_or_x, use_graph_data, edge_attr, "edge_attr", "NNConv")
        row, col = edge_index
        kernels = self.edge_network(edge_attr).view(-1, x.size(-1), self.out_channels)
        messages = torch.bmm(x[row].unsqueeze(1), kernels).squeeze(1)
        out = torch.zeros(x.size(0), self.out_channels, dtype=x.dtype, device=x.device)
        out.index_add_(0, col, messages)
        if self.aggr == "mean":
            degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            degree.index_add_(0, col, torch.ones(col.size(0), dtype=x.dtype, device=x.device))
            out = out / degree.clamp_min(1).unsqueeze(-1)
        return out + self.root_linear(x)


class ECConv(NNConv):
    pass
