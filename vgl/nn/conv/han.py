import torch
from torch import nn

from vgl.nn.conv.rgcn import _relation_key


class HANConv(nn.Module):
    def __init__(self, in_channels, out_channels, *, node_types, relation_types):
        super().__init__()
        self.out_channels = out_channels
        self.node_types = tuple(node_types)
        self.relation_types = tuple(relation_types)

        self.self_linears = nn.ModuleDict(
            {node_type: nn.Linear(in_channels, out_channels) for node_type in self.node_types}
        )
        self.relation_linears = nn.ModuleDict(
            {
                _relation_key(edge_type): nn.Linear(in_channels, out_channels, bias=False)
                for edge_type in self.relation_types
            }
        )
        self.semantic_linear = nn.Linear(out_channels, out_channels)
        self.semantic_score = nn.Linear(out_channels, 1, bias=False)

    def forward(self, graph):
        relation_outputs = {
            node_type: [self.self_linears[node_type](graph.nodes[node_type].x)]
            for node_type in self.node_types
        }

        for edge_type in self.relation_types:
            if edge_type not in graph.edges:
                continue
            src_type, _, dst_type = edge_type
            row, col = graph.edges[edge_type].edge_index
            src_x = graph.nodes[src_type].x
            transformed = self.relation_linears[_relation_key(edge_type)](src_x)
            out = torch.zeros(
                graph.nodes[dst_type].x.size(0),
                self.out_channels,
                dtype=src_x.dtype,
                device=src_x.device,
            )
            out.index_add_(0, col, transformed[row])
            degree = torch.zeros(graph.nodes[dst_type].x.size(0), dtype=src_x.dtype, device=src_x.device)
            degree.index_add_(0, col, torch.ones(col.size(0), dtype=src_x.dtype, device=src_x.device))
            relation_outputs[dst_type].append(out / degree.clamp_min(1).unsqueeze(-1))

        outputs = {}
        for node_type, candidates in relation_outputs.items():
            stacked = torch.stack(candidates, dim=1)
            scores = self.semantic_score(torch.tanh(self.semantic_linear(stacked))).squeeze(-1)
            weights = torch.softmax(scores, dim=1).unsqueeze(-1)
            outputs[node_type] = (stacked * weights).sum(dim=1)
        return outputs
