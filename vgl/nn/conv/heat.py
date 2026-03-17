import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import edge_softmax
from vgl.nn.conv.rgcn import _relation_key


def _resolve_relation_edge_attr(graph, edge_type, edge_channels, layer_name):
    if edge_channels <= 0:
        return None
    store = graph.edges[edge_type]
    if "edge_attr" not in store.data:
        raise ValueError(f"{layer_name} requires edge_attr for relation {edge_type!r}")
    return store.data["edge_attr"]


class HEATConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        node_types,
        relation_types,
        edge_channels,
        heads=1,
        dropout=0.0,
        root_weight=True,
        bias=True,
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("HEATConv requires heads >= 1")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("HEATConv requires dropout to be in [0, 1]")

        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.node_types = tuple(node_types)
        self.relation_types = tuple(relation_types)
        self.edge_channels = edge_channels

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
        self.node_type_embeddings = nn.ParameterDict(
            {node_type: nn.Parameter(torch.randn(hidden_channels)) for node_type in self.node_types}
        )
        self.relation_embeddings = nn.ParameterDict(
            {_relation_key(edge_type): nn.Parameter(torch.randn(hidden_channels)) for edge_type in self.relation_types}
        )
        self.edge_linear = nn.Linear(edge_channels, hidden_channels, bias=False) if edge_channels > 0 else None
        self.attn_linear = nn.Linear(out_channels * 4, 1, bias=False)
        self.output_linears = nn.ModuleDict(
            {node_type: nn.Linear(hidden_channels, out_channels, bias=bias) for node_type in self.node_types}
        )
        self.root_linears = (
            nn.ModuleDict({node_type: nn.Linear(in_channels, out_channels, bias=False) for node_type in self.node_types})
            if root_weight
            else None
        )

    def forward(self, graph):
        queries = {}
        keys = {}
        values = {}
        outputs = {}

        for node_type in self.node_types:
            x = graph.nodes[node_type].x
            queries[node_type] = self.query_linears[node_type](x).view(-1, self.heads, self.out_channels)
            keys[node_type] = self.key_linears[node_type](x).view(-1, self.heads, self.out_channels)
            values[node_type] = self.value_linears[node_type](x).view(-1, self.heads, self.out_channels)
            outputs[node_type] = torch.zeros(
                x.size(0),
                self.heads * self.out_channels,
                dtype=x.dtype,
                device=x.device,
            )

        for edge_type in self.relation_types:
            if edge_type not in graph.edges:
                continue
            src_type, _, dst_type = edge_type
            row, col = graph.edges[edge_type].edge_index
            num_dst = graph.nodes[dst_type].x.size(0)
            relation_name = _relation_key(edge_type)

            src_bias = self.node_type_embeddings[src_type].view(1, self.heads, self.out_channels)
            dst_bias = self.node_type_embeddings[dst_type].view(1, self.heads, self.out_channels)
            relation_bias = self.relation_embeddings[relation_name].view(1, self.heads, self.out_channels)

            if self.edge_linear is not None:
                edge_attr = _resolve_relation_edge_attr(graph, edge_type, self.edge_channels, "HEATConv")
                edge_bias = self.edge_linear(edge_attr).view(-1, self.heads, self.out_channels)
            else:
                edge_bias = torch.zeros(
                    row.size(0),
                    self.heads,
                    self.out_channels,
                    dtype=graph.nodes[src_type].x.dtype,
                    device=graph.nodes[src_type].x.device,
                )

            attention_inputs = torch.cat(
                [
                    queries[dst_type][col] + dst_bias,
                    keys[src_type][row] + src_bias,
                    relation_bias.expand(row.size(0), -1, -1),
                    edge_bias,
                ],
                dim=-1,
            )
            scores = self.attn_linear(torch.tanh(attention_inputs.reshape(-1, self.out_channels * 4))).view(
                -1,
                self.heads,
            )
            weights = torch.stack(
                [edge_softmax(scores[:, head], graph.edges[edge_type].edge_index, num_dst) for head in range(self.heads)],
                dim=-1,
            )
            weights = F.dropout(weights, p=self.dropout, training=self.training)

            messages = values[src_type][row] + relation_bias + edge_bias
            aggregated = torch.zeros(
                num_dst,
                self.heads,
                self.out_channels,
                dtype=graph.nodes[dst_type].x.dtype,
                device=graph.nodes[dst_type].x.device,
            )
            aggregated.index_add_(0, col, messages * weights.unsqueeze(-1))
            outputs[dst_type] += aggregated.reshape(num_dst, self.heads * self.out_channels)

        final_outputs = {}
        for node_type in self.node_types:
            out = self.output_linears[node_type](outputs[node_type])
            if self.root_linears is not None:
                out = out + self.root_linears[node_type](graph.nodes[node_type].x)
            final_outputs[node_type] = out

        return final_outputs
