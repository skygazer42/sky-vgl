import math

import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class DNAConv(nn.Module):
    def __init__(self, channels, heads=1, dropout=0.0):
        super().__init__()
        if heads < 1:
            raise ValueError("DNAConv requires heads >= 1")
        if channels % heads != 0:
            raise ValueError("DNAConv requires channels divisible by heads")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("DNAConv requires dropout to be in [0, 1]")

        self.channels = channels
        self.out_channels = channels
        self.heads = heads
        self.dropout = dropout
        self.head_dim = channels // heads

        self.query_linear = nn.Linear(channels, channels)
        self.key_linear = nn.Linear(channels, channels)
        self.value_linear = nn.Linear(channels, channels)
        self.output_linear = nn.Linear(channels, channels)

    def _coerce_history(self, graph_or_x, edge_index, history):
        if edge_index is None:
            x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "DNAConv")
            history = x.unsqueeze(1) if history is None else history
        else:
            history = graph_or_x if history is None else history

        if history.dim() == 2:
            history = history.unsqueeze(1)
        if history.dim() != 3:
            raise ValueError("DNAConv requires history with shape [num_nodes, num_layers, channels]")
        if history.size(-1) != self.channels:
            raise ValueError("DNAConv history channel dimension must match channels")
        return history, edge_index

    def forward(self, graph_or_x, edge_index=None, history=None):
        history, edge_index = self._coerce_history(graph_or_x, edge_index, history)
        row, col = edge_index
        current = history[:, -1]
        num_nodes, num_layers, _ = history.shape

        query = self.query_linear(current).view(num_nodes, self.heads, self.head_dim)
        key = self.key_linear(history).view(num_nodes, num_layers, self.heads, self.head_dim)
        value = self.value_linear(history).view(num_nodes, num_layers, self.heads, self.head_dim)

        query_dst = query[col].unsqueeze(1)
        key_src = key[row]
        scores = (query_dst * key_src).sum(dim=-1) / math.sqrt(self.head_dim)
        token_weights = F.softmax(scores, dim=1)
        token_weights = F.dropout(token_weights, p=self.dropout, training=self.training)
        edge_messages = (token_weights.unsqueeze(-1) * value[row]).sum(dim=1)

        edge_scores = scores.max(dim=1).values
        edge_weights = torch.stack(
            [edge_softmax(edge_scores[:, head], edge_index, num_nodes) for head in range(self.heads)],
            dim=-1,
        )
        edge_weights = F.dropout(edge_weights, p=self.dropout, training=self.training)

        out = torch.zeros(num_nodes, self.heads, self.head_dim, dtype=current.dtype, device=current.device)
        out.index_add_(0, col, edge_messages * edge_weights.unsqueeze(-1))
        out = out.reshape(num_nodes, self.channels)
        return self.output_linear(out) + current
