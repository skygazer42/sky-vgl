import math

import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import edge_softmax
from vgl.nn.conv.rgcn import _relation_key


class HGTConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        node_types,
        relation_types,
        heads=1,
        dropout=0.0,
        root_weight=True,
        bias=True,
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("HGTConv requires heads >= 1")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("HGTConv requires dropout to be in [0, 1]")

        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.node_types = tuple(node_types)
        self.relation_types = tuple(relation_types)
        self.root_weight = root_weight

        hidden_channels = heads * out_channels
        self.query_linears = nn.ModuleDict(
            {node_type: nn.Linear(in_channels, hidden_channels, bias=False) for node_type in self.node_types}
        )
        self.key_linears = nn.ModuleDict(
            {node_type: nn.Linear(in_channels, hidden_channels, bias=False) for node_type in self.node_types}
        )
        self.value_linears = nn.ModuleDict(
            {node_type: nn.Linear(in_channels, hidden_channels, bias=False) for node_type in self.node_types}
        )
        self.relation_key_linears = nn.ModuleDict(
            {
                _relation_key(edge_type): nn.Linear(hidden_channels, hidden_channels, bias=False)
                for edge_type in self.relation_types
            }
        )
        self.relation_value_linears = nn.ModuleDict(
            {
                _relation_key(edge_type): nn.Linear(hidden_channels, hidden_channels, bias=False)
                for edge_type in self.relation_types
            }
        )
        self.output_linears = nn.ModuleDict(
            {node_type: nn.Linear(hidden_channels, out_channels, bias=bias) for node_type in self.node_types}
        )
        self.root_linears = (
            nn.ModuleDict(
                {node_type: nn.Linear(in_channels, out_channels, bias=False) for node_type in self.node_types}
            )
            if root_weight
            else None
        )

    def forward(self, graph):
        hidden_channels = self.heads * self.out_channels
        queries = {}
        keys = {}
        values = {}
        outputs = {}

        for node_type in self.node_types:
            x = graph.nodes[node_type].x
            queries[node_type] = self.query_linears[node_type](x).view(-1, self.heads, self.out_channels)
            keys[node_type] = self.key_linears[node_type](x)
            values[node_type] = self.value_linears[node_type](x)
            outputs[node_type] = torch.zeros(
                x.size(0),
                hidden_channels,
                dtype=x.dtype,
                device=x.device,
            )

        for edge_type in self.relation_types:
            if edge_type not in graph.edges:
                continue
            src_type, _, dst_type = edge_type
            relation_name = _relation_key(edge_type)
            row, col = graph.edges[edge_type].edge_index
            num_dst = graph.nodes[dst_type].x.size(0)

            relation_keys = self.relation_key_linears[relation_name](keys[src_type]).view(
                -1,
                self.heads,
                self.out_channels,
            )
            relation_values = self.relation_value_linears[relation_name](values[src_type]).view(
                -1,
                self.heads,
                self.out_channels,
            )
            scores = (queries[dst_type][col] * relation_keys[row]).sum(dim=-1) / math.sqrt(self.out_channels)
            weights = torch.stack(
                [edge_softmax(scores[:, head], graph.edges[edge_type].edge_index, num_dst) for head in range(self.heads)],
                dim=-1,
            )
            weights = F.dropout(weights, p=self.dropout, training=self.training)

            aggregated = torch.zeros(
                num_dst,
                self.heads,
                self.out_channels,
                dtype=graph.nodes[dst_type].x.dtype,
                device=graph.nodes[dst_type].x.device,
            )
            aggregated.index_add_(0, col, relation_values[row] * weights.unsqueeze(-1))
            outputs[dst_type] += aggregated.reshape(num_dst, hidden_channels)

        final_outputs = {}
        for node_type in self.node_types:
            out = self.output_linears[node_type](outputs[node_type])
            if self.root_linears is not None:
                out = out + self.root_linears[node_type](graph.nodes[node_type].x)
            final_outputs[node_type] = out

        return final_outputs
