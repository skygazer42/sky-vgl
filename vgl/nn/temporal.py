import math

import torch
import torch.nn.functional as F
from torch import nn

from vgl.nn.conv._homo import coerce_homo_inputs, edge_softmax


def _resolve_edge_time(graph_or_x, use_graph_data, edge_time, layer_name):
    if edge_time is not None:
        return edge_time
    if not use_graph_data:
        raise ValueError(f"{layer_name} requires edge_time when called with x and edge_index")
    time_attr = graph_or_x.schema.time_attr
    if time_attr is None or time_attr not in graph_or_x.edata:
        raise ValueError(f"{layer_name} requires a temporal graph with edge timestamps")
    return graph_or_x.edata[time_attr]


def _edge_query_time(query_time, num_nodes, num_edges, col, dtype, device, layer_name):
    if query_time is None:
        raise ValueError(f"{layer_name} requires query_time")

    query_time = torch.as_tensor(query_time, dtype=dtype, device=device)
    if query_time.ndim == 0 or query_time.numel() == 1:
        return query_time.reshape(1).expand(num_edges)
    if query_time.ndim == 1 and query_time.size(0) == num_nodes:
        return query_time[col]
    if query_time.ndim == 1 and query_time.size(0) == num_edges:
        return query_time
    raise ValueError(f"{layer_name} query_time must be a scalar, per-node tensor, or per-edge tensor")


def _feed_forward(channels, ff_multiplier, dropout):
    hidden_channels = channels * ff_multiplier
    return nn.Sequential(
        nn.Linear(channels, hidden_channels),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_channels, channels),
    )


class TimeEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.linear = nn.Linear(1, out_channels)

    def forward(self, t):
        t = torch.as_tensor(t, dtype=self.linear.weight.dtype, device=self.linear.weight.device)
        return torch.cos(self.linear(t.reshape(-1, 1)))


class IdentityTemporalMessage(nn.Module):
    def __init__(self, *, memory_channels, raw_message_channels, time_channels):
        super().__init__()
        self.memory_channels = memory_channels
        self.raw_message_channels = raw_message_channels
        self.time_channels = time_channels
        self.time_encoder = TimeEncoder(time_channels)
        self.out_channels = (2 * memory_channels) + raw_message_channels + time_channels

    def forward(self, src_memory, dst_memory, raw_message, delta_time):
        if src_memory.shape != dst_memory.shape:
            raise ValueError("IdentityTemporalMessage requires src_memory and dst_memory with matching shapes")
        if src_memory.ndim != 2:
            raise ValueError("IdentityTemporalMessage requires 2D memory tensors")

        if raw_message is None:
            raw_message = src_memory.new_zeros(src_memory.size(0), self.raw_message_channels)
        else:
            raw_message = torch.as_tensor(raw_message, dtype=src_memory.dtype, device=src_memory.device)
            if raw_message.ndim == 1:
                raw_message = raw_message.reshape(1, -1)
            if raw_message.ndim != 2 or raw_message.size(0) != src_memory.size(0):
                raise ValueError("IdentityTemporalMessage raw_message must align with the event dimension")
            if raw_message.size(-1) != self.raw_message_channels:
                raise ValueError("IdentityTemporalMessage raw_message has unexpected feature size")

        time_encoding = self.time_encoder(delta_time.to(dtype=src_memory.dtype, device=src_memory.device))
        return torch.cat([src_memory, dst_memory, raw_message, time_encoding], dim=-1)


class LastMessageAggregator(nn.Module):
    def forward(self, messages, node_index, timestamp, num_nodes=None):
        del num_nodes
        if node_index.numel() == 0:
            return (
                node_index.new_empty(0),
                messages.new_empty(0, messages.size(-1)),
                timestamp.new_empty(0),
            )

        node_ids = torch.unique(node_index, sorted=True)
        aggregated_messages = []
        aggregated_timestamps = []
        for node_id in node_ids.tolist():
            mask = node_index == node_id
            node_timestamps = timestamp[mask]
            latest_idx = torch.argmax(node_timestamps)
            aggregated_messages.append(messages[mask][latest_idx])
            aggregated_timestamps.append(node_timestamps[latest_idx])
        return node_ids, torch.stack(aggregated_messages, dim=0), torch.stack(aggregated_timestamps, dim=0)


class MeanMessageAggregator(nn.Module):
    def forward(self, messages, node_index, timestamp, num_nodes=None):
        del num_nodes
        if node_index.numel() == 0:
            return (
                node_index.new_empty(0),
                messages.new_empty(0, messages.size(-1)),
                timestamp.new_empty(0),
            )

        node_ids = torch.unique(node_index, sorted=True)
        aggregated_messages = []
        aggregated_timestamps = []
        for node_id in node_ids.tolist():
            mask = node_index == node_id
            aggregated_messages.append(messages[mask].mean(dim=0))
            aggregated_timestamps.append(timestamp[mask].max())
        return node_ids, torch.stack(aggregated_messages, dim=0), torch.stack(aggregated_timestamps, dim=0)


class TGNMemory(nn.Module):
    def __init__(
        self,
        *,
        num_nodes,
        memory_channels,
        raw_message_channels,
        time_channels,
        message_module=None,
        aggregator="last",
    ):
        super().__init__()
        if num_nodes < 1:
            raise ValueError("TGNMemory requires num_nodes >= 1")
        if memory_channels < 1:
            raise ValueError("TGNMemory requires memory_channels >= 1")
        if raw_message_channels < 0:
            raise ValueError("TGNMemory requires raw_message_channels >= 0")
        if time_channels < 1:
            raise ValueError("TGNMemory requires time_channels >= 1")

        self.num_nodes = num_nodes
        self.memory_channels = memory_channels
        self.raw_message_channels = raw_message_channels
        self.time_channels = time_channels
        self.message_module = message_module or IdentityTemporalMessage(
            memory_channels=memory_channels,
            raw_message_channels=raw_message_channels,
            time_channels=time_channels,
        )
        if not hasattr(self.message_module, "out_channels"):
            raise ValueError("TGNMemory message_module must expose out_channels")

        if aggregator == "last":
            self.aggregator = LastMessageAggregator()
        elif aggregator == "mean":
            self.aggregator = MeanMessageAggregator()
        elif isinstance(aggregator, nn.Module):
            self.aggregator = aggregator
        else:
            raise ValueError("TGNMemory aggregator must be 'last', 'mean', or an nn.Module")

        self.memory_updater = nn.GRUCell(self.message_module.out_channels, memory_channels)
        self.register_buffer("memory", torch.zeros(num_nodes, memory_channels))
        self.register_buffer("last_update", torch.zeros(num_nodes, dtype=torch.long))

    def reset_state(self):
        self.memory = torch.zeros_like(self.memory)
        self.last_update = torch.zeros_like(self.last_update)

    def detach(self):
        self.memory = self.memory.detach()

    def forward(self, node_ids=None):
        if node_ids is None:
            return self.memory
        node_ids = torch.as_tensor(node_ids, dtype=torch.long, device=self.memory.device)
        return self.memory[node_ids]

    def update(self, *, src_index, dst_index, timestamp, raw_message=None):
        src_index = torch.as_tensor(src_index, dtype=torch.long, device=self.memory.device)
        dst_index = torch.as_tensor(dst_index, dtype=torch.long, device=self.memory.device)
        timestamp = torch.as_tensor(timestamp, dtype=torch.long, device=self.memory.device)

        if src_index.ndim != 1 or dst_index.ndim != 1 or timestamp.ndim != 1:
            raise ValueError("TGNMemory update expects 1D src_index, dst_index, and timestamp tensors")
        if not (src_index.numel() == dst_index.numel() == timestamp.numel()):
            raise ValueError("TGNMemory update tensors must share the same length")

        if raw_message is None:
            raw_message = self.memory.new_zeros(src_index.size(0), self.raw_message_channels)
        else:
            raw_message = torch.as_tensor(raw_message, dtype=self.memory.dtype, device=self.memory.device)
            if raw_message.ndim == 1:
                raw_message = raw_message.reshape(1, -1)
            if raw_message.ndim != 2 or raw_message.size(0) != src_index.size(0):
                raise ValueError("TGNMemory raw_message must align with the event dimension")
            if raw_message.size(-1) != self.raw_message_channels:
                raise ValueError("TGNMemory raw_message has unexpected feature size")

        src_memory = self.memory[src_index]
        dst_memory = self.memory[dst_index]
        delta_src = timestamp.to(dtype=self.memory.dtype) - self.last_update[src_index].to(dtype=self.memory.dtype)
        delta_dst = timestamp.to(dtype=self.memory.dtype) - self.last_update[dst_index].to(dtype=self.memory.dtype)

        src_messages = self.message_module(src_memory, dst_memory, raw_message, delta_src)
        dst_messages = self.message_module(dst_memory, src_memory, raw_message, delta_dst)
        node_index = torch.cat([src_index, dst_index], dim=0)
        messages = torch.cat([src_messages, dst_messages], dim=0)
        message_timestamp = torch.cat([timestamp, timestamp], dim=0)

        node_ids, aggregated_messages, aggregated_timestamps = self.aggregator(
            messages=messages,
            node_index=node_index,
            timestamp=message_timestamp,
            num_nodes=self.num_nodes,
        )
        updated_memory = self.memory_updater(aggregated_messages, self.memory[node_ids])

        memory = self.memory.clone()
        memory[node_ids] = updated_memory
        self.memory = memory

        last_update = self.last_update.clone()
        last_update[node_ids] = aggregated_timestamps.to(dtype=self.last_update.dtype)
        self.last_update = last_update

        return self.memory[node_ids]


class TGATLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        time_channels,
        heads=1,
        dropout=0.0,
        root_weight=True,
    ):
        super().__init__()
        if heads < 1:
            raise ValueError("TGATLayer requires heads >= 1")
        if dropout < 0.0 or dropout > 1.0:
            raise ValueError("TGATLayer requires dropout to be in [0, 1]")

        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.root_weight = root_weight

        hidden_channels = heads * out_channels
        self.query_linear = nn.Linear(in_channels, hidden_channels, bias=False)
        self.key_linear = nn.Linear(in_channels, hidden_channels, bias=False)
        self.value_linear = nn.Linear(in_channels, hidden_channels, bias=False)
        self.time_encoder = TimeEncoder(time_channels)
        self.time_linear = nn.Linear(time_channels, hidden_channels, bias=False)
        self.output_linear = nn.Linear(hidden_channels, out_channels)
        self.root_linear = nn.Linear(in_channels, out_channels, bias=False) if root_weight else None

    def forward(self, graph_or_x, edge_index=None, *, edge_time=None, query_time=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "TGATLayer")
        edge_time = _resolve_edge_time(graph_or_x, use_graph_data, edge_time, "TGATLayer").to(
            dtype=x.dtype,
            device=x.device,
        )
        row, col = edge_index
        query_time = _edge_query_time(
            query_time,
            x.size(0),
            row.size(0),
            col,
            x.dtype,
            x.device,
            "TGATLayer",
        )
        delta = query_time - edge_time
        time_bias = self.time_linear(self.time_encoder(delta)).view(-1, self.heads, self.out_channels)

        query = self.query_linear(x).view(x.size(0), self.heads, self.out_channels)
        key = self.key_linear(x).view(x.size(0), self.heads, self.out_channels)
        value = self.value_linear(x).view(x.size(0), self.heads, self.out_channels)

        scores = (query[col] * (key[row] + time_bias)).sum(dim=-1) / math.sqrt(self.out_channels)
        weights = torch.stack(
            [edge_softmax(scores[:, head], edge_index, x.size(0)) for head in range(self.heads)],
            dim=-1,
        )
        weights = F.dropout(weights, p=self.dropout, training=self.training)

        out = torch.zeros(
            x.size(0),
            self.heads,
            self.out_channels,
            dtype=x.dtype,
            device=x.device,
        )
        out.index_add_(0, col, (value[row] + time_bias) * weights.unsqueeze(-1))
        out = self.output_linear(out.reshape(x.size(0), self.heads * self.out_channels))
        if self.root_linear is not None:
            out = out + self.root_linear(x)
        return out


class TGATEncoder(nn.Module):
    def __init__(
        self,
        *,
        channels,
        num_layers,
        time_channels,
        heads=1,
        dropout=0.0,
        ff_multiplier=2,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("TGATEncoder requires num_layers >= 1")

        self.channels = channels
        self.out_channels = channels
        self.layers = nn.ModuleList(
            [
                TGATLayer(
                    in_channels=channels,
                    out_channels=channels,
                    time_channels=time_channels,
                    heads=heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(channels) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([_feed_forward(channels, ff_multiplier, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_or_x, edge_index=None, *, edge_time=None, query_time=None):
        use_graph_data = edge_index is None
        x, edge_index = coerce_homo_inputs(graph_or_x, edge_index, "TGATEncoder")
        edge_time = _resolve_edge_time(graph_or_x, use_graph_data, edge_time, "TGATEncoder")

        for layer, norm, ffn in zip(self.layers, self.norms, self.ffns):
            attended = layer(x, edge_index, edge_time=edge_time, query_time=query_time)
            x = norm(x + self.dropout(attended))
            x = norm(x + self.dropout(ffn(x)))
        return x
