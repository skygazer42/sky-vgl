import math

import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, mean_propagate, node_degree, propagate_steps
from vgl.nn.conv.transformer import TransformerConv


def _feed_forward(channels, ff_multiplier, dropout):
    hidden_channels = channels * ff_multiplier
    return nn.Sequential(
        nn.Linear(channels, hidden_channels),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_channels, channels),
    )


def _shortest_path_buckets(edge_index, num_nodes, max_distance, device):
    far_bucket = max_distance + 1
    distances = torch.full((num_nodes, num_nodes), far_bucket, dtype=torch.long, device=device)
    if num_nodes == 0:
        return distances

    node_ids = torch.arange(num_nodes, dtype=torch.long, device=device)
    distances[node_ids, node_ids] = 0
    if edge_index.numel() == 0 or max_distance <= 0:
        return distances

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    row = edge_index[0].to(dtype=torch.long, device=device)
    col = edge_index[1].to(dtype=torch.long, device=device)
    adjacency[row, col] = True
    adjacency[col, row] = True

    seen = torch.eye(num_nodes, dtype=torch.bool, device=device)
    frontier = adjacency
    adjacency_int = adjacency.to(dtype=torch.int32)
    for distance in range(1, max_distance + 1):
        frontier = frontier & ~seen
        distances[frontier] = distance
        seen |= frontier
        frontier = (frontier.to(dtype=torch.int32) @ adjacency_int) > 0

    return distances


class GraphTransformerEncoderLayer(nn.Module):
    def __init__(self, *, channels, heads=1, dropout=0.0, ff_multiplier=2, beta=False):
        super().__init__()
        self.channels = channels
        self.out_channels = channels
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.attention = TransformerConv(
            in_channels=channels,
            out_channels=channels,
            heads=heads,
            concat=False,
            beta=beta,
            dropout=dropout,
        )
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = _feed_forward(channels, ff_multiplier, dropout)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GraphTransformerEncoderLayer")
        attended = self.attention(x, edge_index)
        x = self.norm1(x + self.dropout(attended))
        feed_forward = self.ffn(x)
        return self.norm2(x + self.dropout(feed_forward))


class GraphTransformerEncoder(nn.Module):
    def __init__(self, *, channels, num_layers, heads=1, dropout=0.0, ff_multiplier=2, beta=False):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GraphTransformerEncoder requires num_layers >= 1")

        self.channels = channels
        self.out_channels = channels
        self.layers = nn.ModuleList(
            [
                GraphTransformerEncoderLayer(
                    channels=channels,
                    heads=heads,
                    dropout=dropout,
                    ff_multiplier=ff_multiplier,
                    beta=beta,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GraphTransformerEncoder")
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GraphormerEncoderLayer(nn.Module):
    def __init__(self, *, channels, heads=1, max_distance=4, max_degree=32, dropout=0.0, ff_multiplier=2):
        super().__init__()
        if heads < 1:
            raise ValueError("GraphormerEncoderLayer requires heads >= 1")
        if channels % heads != 0:
            raise ValueError("GraphormerEncoderLayer requires channels divisible by heads")

        self.channels = channels
        self.out_channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.max_distance = max_distance
        self.max_degree = max_degree
        self.dropout = nn.Dropout(dropout)

        self.degree_embedding = nn.Embedding(max_degree + 2, channels)
        self.spatial_bias = nn.Embedding(max_distance + 2, heads)
        self.query_linear = nn.Linear(channels, channels)
        self.key_linear = nn.Linear(channels, channels)
        self.value_linear = nn.Linear(channels, channels)
        self.output_linear = nn.Linear(channels, channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = _feed_forward(channels, ff_multiplier, dropout)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GraphormerEncoderLayer")
        num_nodes = x.size(0)
        degree = node_degree(edge_index, num_nodes, x.dtype, x.device).long().clamp(max=self.max_degree + 1)
        hidden = x + self.degree_embedding(degree)

        query = self.query_linear(hidden).view(num_nodes, self.heads, self.head_dim).permute(1, 0, 2)
        key = self.key_linear(hidden).view(num_nodes, self.heads, self.head_dim).permute(1, 0, 2)
        value = self.value_linear(hidden).view(num_nodes, self.heads, self.head_dim).permute(1, 0, 2)

        scores = torch.einsum("hnd,hmd->hnm", query, key) / math.sqrt(self.head_dim)
        distances = _shortest_path_buckets(edge_index, num_nodes, self.max_distance, x.device)
        bias = self.spatial_bias(distances).permute(2, 0, 1)
        attention = F.softmax(scores + bias, dim=-1)
        attention = self.dropout(attention)
        attended = torch.einsum("hnm,hmd->hnd", attention, value).permute(1, 0, 2).reshape(num_nodes, self.channels)
        attended = self.output_linear(attended)

        hidden = self.norm1(x + self.dropout(attended))
        feed_forward = self.ffn(hidden)
        return self.norm2(hidden + self.dropout(feed_forward))


class GraphormerEncoder(nn.Module):
    def __init__(
        self,
        *,
        channels,
        num_layers,
        heads=1,
        max_distance=4,
        max_degree=32,
        dropout=0.0,
        ff_multiplier=2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("GraphormerEncoder requires num_layers >= 1")

        self.channels = channels
        self.out_channels = channels
        self.layers = nn.ModuleList(
            [
                GraphormerEncoderLayer(
                    channels=channels,
                    heads=heads,
                    max_distance=max_distance,
                    max_degree=max_degree,
                    dropout=dropout,
                    ff_multiplier=ff_multiplier,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GraphormerEncoder")
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class GPSLayer(nn.Module):
    def __init__(self, *, channels, local_gnn, heads=1, dropout=0.0, ff_multiplier=2):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("GPSLayer requires channels divisible by heads")

        self.channels = channels
        self.out_channels = channels
        self.local_gnn = local_gnn
        self.global_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = _feed_forward(channels, ff_multiplier, dropout)

    def _run_local(self, graph_or_x, x, edge_index):
        try:
            return self.local_gnn(x, edge_index)
        except TypeError:
            return self.local_gnn(graph_or_x)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "GPSLayer")
        local_out = self._run_local(graph_or_x, x, edge_index)
        global_out, _ = self.global_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        hidden = self.norm1(x + self.dropout(local_out) + self.dropout(global_out.squeeze(0)))
        feed_forward = self.ffn(hidden)
        return self.norm2(hidden + self.dropout(feed_forward))


class SGFormerEncoderLayer(nn.Module):
    def __init__(self, *, channels, heads=1, alpha=0.5, dropout=0.0, ff_multiplier=2):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("SGFormerEncoderLayer requires channels divisible by heads")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("SGFormerEncoderLayer requires alpha in [0, 1]")

        self.channels = channels
        self.out_channels = channels
        self.alpha = alpha
        self.local_linear = nn.Linear(channels, channels)
        self.global_attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_linear = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = _feed_forward(channels, ff_multiplier, dropout)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "SGFormerEncoderLayer")
        local_out = mean_propagate(self.local_linear(x), edge_index)
        global_out, _ = self.global_attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        mixed = self.alpha * local_out + (1.0 - self.alpha) * global_out.squeeze(0)
        hidden = self.norm1(x + self.dropout(self.output_linear(mixed)))
        feed_forward = self.ffn(hidden)
        return self.norm2(hidden + self.dropout(feed_forward))


class SGFormerEncoder(nn.Module):
    def __init__(self, *, channels, num_layers, heads=1, alpha=0.5, dropout=0.0, ff_multiplier=2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("SGFormerEncoder requires num_layers >= 1")

        self.channels = channels
        self.out_channels = channels
        self.layers = nn.ModuleList(
            [
                SGFormerEncoderLayer(
                    channels=channels,
                    heads=heads,
                    alpha=alpha,
                    dropout=dropout,
                    ff_multiplier=ff_multiplier,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "SGFormerEncoder")
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class NAGphormerEncoder(nn.Module):
    def __init__(self, *, channels, num_layers, num_hops, heads=1, dropout=0.0, ff_multiplier=2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("NAGphormerEncoder requires num_layers >= 1")
        if channels % heads != 0:
            raise ValueError("NAGphormerEncoder requires channels divisible by heads")
        if num_hops < 1:
            raise ValueError("NAGphormerEncoder requires num_hops >= 1")

        self.channels = channels
        self.out_channels = channels
        self.num_hops = num_hops
        self.hop_embedding = nn.Embedding(num_hops + 1, channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, graph_or_x, edge_index=None):
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "NAGphormerEncoder")
        tokens = torch.stack(propagate_steps(x, edge_index, self.num_hops), dim=1)
        hop_index = torch.arange(self.num_hops + 1, device=x.device)
        tokens = tokens + self.hop_embedding(hop_index).unsqueeze(0)
        encoded = self.encoder(tokens)
        pooled = encoded.mean(dim=1)
        return self.norm(pooled + x)
