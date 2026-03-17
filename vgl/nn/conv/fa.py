import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


class FAConv(nn.Module):
    def __init__(self, channels, eps=0.1, dropout=0.0):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.eps = eps
        self.dropout = dropout
        self.attention = nn.Linear(channels * 2, 1)
        self.message_linear = nn.Linear(channels, channels, bias=False)

    def forward(self, graph_or_x, edge_index=None, x0=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "FAConv")
        if x0 is None:
            x0 = x

        row, col = edge_index
        pair = torch.cat([x[row], x[col]], dim=-1)
        scores = torch.tanh(self.attention(pair)).squeeze(-1)
        weights = edge_softmax(scores, edge_index, x.size(0))
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        messages = self.message_linear(x[row]) * weights.unsqueeze(-1)
        out = torch.zeros_like(x)
        out.index_add_(0, col, messages)
        return (1.0 - self.eps) * out + self.eps * x0
