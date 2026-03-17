import torch
from torch import nn


def _relation_key(edge_type):
    return "__".join(edge_type)


class RGCNConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        node_types,
        relation_types,
        aggr="mean",
        root_weight=True,
        bias=True,
    ):
        super().__init__()
        if aggr not in {"mean", "sum"}:
            raise ValueError("RGCNConv only supports mean and sum aggregation")

        self.out_channels = out_channels
        self.node_types = tuple(node_types)
        self.relation_types = tuple(relation_types)
        self.aggr = aggr
        self.root_weight = root_weight

        self.relation_linears = nn.ModuleDict(
            {
                _relation_key(edge_type): nn.Linear(in_channels, out_channels, bias=False)
                for edge_type in self.relation_types
            }
        )
        self.root_linears = (
            nn.ModuleDict(
                {
                    node_type: nn.Linear(in_channels, out_channels, bias=bias)
                    for node_type in self.node_types
                }
            )
            if root_weight
            else None
        )

    def forward(self, graph):
        aggregated_outputs = {}
        counts = {}
        for node_type in self.node_types:
            x = graph.nodes[node_type].x
            aggregated_outputs[node_type] = torch.zeros(
                x.size(0),
                self.out_channels,
                dtype=x.dtype,
                device=x.device,
            )
            counts[node_type] = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)

        for edge_type in self.relation_types:
            if edge_type not in graph.edges:
                continue
            src_type, _, dst_type = edge_type
            row, col = graph.edges[edge_type].edge_index
            src_x = graph.nodes[src_type].x
            messages = self.relation_linears[_relation_key(edge_type)](src_x)[row]
            aggregated_outputs[dst_type].index_add_(0, col, messages)
            counts[dst_type].index_add_(
                0,
                col,
                torch.ones(col.size(0), dtype=src_x.dtype, device=src_x.device),
            )

        if self.aggr == "mean":
            for node_type in self.node_types:
                aggregated_outputs[node_type] = (
                    aggregated_outputs[node_type] / counts[node_type].clamp_min(1).unsqueeze(-1)
                )

        outputs = {}
        for node_type in self.node_types:
            x = graph.nodes[node_type].x
            outputs[node_type] = aggregated_outputs[node_type]
            if self.root_linears is not None:
                outputs[node_type] = outputs[node_type] + self.root_linears[node_type](x)

        return outputs
