from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, SupportsInt, cast

import torch

from vgl._memory import pin_tensor
from vgl.graph.block import Block, HeteroBlock
from vgl.graph.graph import Graph
from vgl.graph.stores import EdgeStore
from vgl.graph.view import GraphView

if TYPE_CHECKING:
    from vgl.dataloading.records import LinkPredictionRecord
    from vgl.dataloading.records import SampleRecord
    from vgl.dataloading.records import TemporalEventRecord


def _expand_interval_values(values: torch.Tensor, counts: torch.Tensor, *, step: int) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    counts = torch.as_tensor(counts, dtype=torch.long, device=values.device).view(-1)
    if values.numel() == 0 or counts.numel() == 0:
        return values.new_empty(0)

    positive = counts > 0
    values = values[positive]
    counts = counts[positive]
    if values.numel() == 0:
        return values.new_empty(0)

    offsets = torch.cumsum(counts, dim=0) - counts
    base = values - step * offsets
    deltas = torch.empty_like(base)
    deltas[0] = base[0]
    if base.numel() > 1:
        deltas[1:] = base[1:] - base[:-1]
    markers = torch.zeros(counts.sum(), dtype=values.dtype, device=values.device)
    markers[offsets] = deltas
    expanded = torch.cumsum(markers, dim=0)
    if step != 0:
        expanded = expanded + step * torch.arange(counts.sum(), dtype=values.dtype, device=values.device)
    return expanded


def _slice_edge_store(store: EdgeStore, mask: torch.Tensor) -> EdgeStore:
    edge_count = int(store.edge_index.size(1))
    edge_data = {}
    for key, value in store.data.items():
        if key == "edge_index":
            edge_data[key] = store.edge_index[:, mask]
        elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[mask]
        else:
            edge_data[key] = value
    return EdgeStore(store.type_name, edge_data)


def _without_supervision_edges(graph, edge_pairs: set[tuple[int, int]]):
    if not edge_pairs:
        return graph
    default_edge_type = graph._default_edge_type()
    return _without_supervision_edges_for_type(graph, default_edge_type, edge_pairs)


def _without_supervision_edges_for_type(graph, edge_type, edge_pairs: set[tuple[int, int]]):
    if not edge_pairs:
        return graph
    edges = {}
    for current_edge_type, store in graph.edges.items():
        if current_edge_type != edge_type:
            edges[current_edge_type] = store
            continue
        edge_index = store.edge_index
        mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        for src_index, dst_index in edge_pairs:
            mask &= ~((edge_index[0] == src_index) & (edge_index[1] == dst_index))
        edges[current_edge_type] = _slice_edge_store(store, mask)
    base = getattr(graph, "base", graph)
    return GraphView(base=base, nodes=graph.nodes, edges=edges, schema=graph.schema)


def _unique_graphs(records):
    graphs = []
    seen = {}
    for record in records:
        graph_id = id(record.graph)
        if graph_id in seen:
            continue
        seen[graph_id] = len(graphs)
        graphs.append(record.graph)
    return graphs


def _unique_blocks(blocks: list[Block | HeteroBlock]) -> list[Block | HeteroBlock]:
    unique = []
    seen = set()
    for block in blocks:
        graph_id = id(block.graph)
        if graph_id in seen:
            continue
        seen.add(graph_id)
        unique.append(block)
    return unique


def _batch_relation_block_layer(blocks: list[Block], *, context: str) -> Block:
    if not blocks:
        raise ValueError(f"{context} block layers require at least one block")
    if len(blocks) == 1:
        return blocks[0]

    first_block = blocks[0]
    for block in blocks[1:]:
        if (
            block.edge_type != first_block.edge_type
            or block.src_type != first_block.src_type
            or block.dst_type != first_block.dst_type
            or block.src_store_type != first_block.src_store_type
            or block.dst_store_type != first_block.dst_store_type
        ):
            raise ValueError(f"{context} requires matching block schemas when batching block layers")

    graphs = [block.graph for block in blocks]
    first_graph = graphs[0]
    if set(first_graph.nodes) == {"node"} and len(first_graph.edges) == 1:
        graph, _ = _batch_homo_graphs(graphs, context=f"{context} blocks")
    else:
        graph, _ = _batch_hetero_graphs(graphs, context=f"{context} blocks")
    return Block(
        graph=graph,
        edge_type=first_block.edge_type,
        src_type=first_block.src_type,
        dst_type=first_block.dst_type,
        src_n_id=torch.cat([block.src_n_id for block in blocks], dim=0),
        dst_n_id=torch.cat([block.dst_n_id for block in blocks], dim=0),
        src_store_type=first_block.src_store_type,
        dst_store_type=first_block.dst_store_type,
    )


def _batch_hetero_block_layer(blocks: list[HeteroBlock], *, context: str) -> HeteroBlock:
    if not blocks:
        raise ValueError(f"{context} block layers require at least one block")
    if len(blocks) == 1:
        return blocks[0]

    first_block = blocks[0]
    for block in blocks[1:]:
        if (
            block.edge_types != first_block.edge_types
            or block.src_store_types != first_block.src_store_types
            or block.dst_store_types != first_block.dst_store_types
            or tuple(block.src_n_id) != tuple(first_block.src_n_id)
            or tuple(block.dst_n_id) != tuple(first_block.dst_n_id)
        ):
            raise ValueError(f"{context} requires matching block schemas when batching block layers")

    graph, _ = _batch_hetero_graphs([block.graph for block in blocks], context=f"{context} blocks")
    return HeteroBlock(
        graph=graph,
        edge_types=first_block.edge_types,
        src_n_id={
            node_type: torch.cat([block.src_n_id[node_type] for block in blocks], dim=0)
            for node_type in first_block.src_n_id
        },
        dst_n_id={
            node_type: torch.cat([block.dst_n_id[node_type] for block in blocks], dim=0)
            for node_type in first_block.dst_n_id
        },
        src_store_types=dict(first_block.src_store_types),
        dst_store_types=dict(first_block.dst_store_types),
    )


def _batch_block_layer(blocks: list[Block | HeteroBlock], *, context: str) -> Block | HeteroBlock:
    if all(isinstance(block, Block) for block in blocks):
        return _batch_relation_block_layer(cast(list[Block], blocks), context=context)
    if all(isinstance(block, HeteroBlock) for block in blocks):
        return _batch_hetero_block_layer(cast(list[HeteroBlock], blocks), context=context)
    raise ValueError(f"{context} requires matching block container types when batching block layers")


def _batch_blocks(records, *, context: str) -> list[Block | HeteroBlock] | None:
    if all(record.blocks is None for record in records):
        return None
    if any(record.blocks is None for record in records):
        raise ValueError(f"{context} requires blocks for every record when any record includes blocks")

    num_layers = len(records[0].blocks)
    if any(len(record.blocks) != num_layers for record in records):
        raise ValueError(f"{context} requires the same number of blocks for every record")

    return [
        _batch_block_layer(_unique_blocks([record.blocks[layer] for record in records]), context=context)
        for layer in range(num_layers)
    ]


def _node_aligned_value(value, count):
    return isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == count


def _node_count(store):
    for value in store.data.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.size(0))
    return 0


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _transfer_tensor(tensor: torch.Tensor, *, device=None, dtype=None, non_blocking: bool = False) -> torch.Tensor:
    can_cast = dtype is not None and (tensor.is_floating_point() or tensor.is_complex())
    if can_cast:
        return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
    return tensor.to(device=device, non_blocking=non_blocking)


def _require_int(value: SupportsInt | None, *, field_name: str) -> int:
    if value is None:
        raise ValueError(f"{field_name} must not be None")
    return int(value)


def _resolve_link_edge_type(record):
    edge_type = getattr(record, "edge_type", None)
    if edge_type is None:
        edge_type = record.metadata.get("edge_type")
    if edge_type is not None:
        return cast(tuple[str, str, str], tuple(edge_type))
    graph = record.graph
    if len(graph.edges) == 1:
        return next(iter(graph.edges))
    try:
        return graph._default_edge_type()
    except AttributeError as exc:
        raise ValueError("Link prediction on heterogeneous graphs requires edge_type") from exc


def _resolve_link_reverse_edge_type(record):
    reverse_edge_type = getattr(record, "reverse_edge_type", None)
    if reverse_edge_type is None:
        reverse_edge_type = record.metadata.get("reverse_edge_type")
    if reverse_edge_type is None:
        return None
    return cast(tuple[str, str, str], tuple(reverse_edge_type))


def _resolve_link_query_id(record):
    query_id = getattr(record, "query_id", None)
    if query_id is None:
        query_id = record.metadata.get("query_id")
    if query_id is None:
        query_id = getattr(record, "sample_id", None)
    if query_id is None:
        query_id = record.metadata.get("sample_id")
    return query_id


def _resolve_temporal_edge_type(record):
    edge_type = getattr(record, "edge_type", None)
    if edge_type is None:
        edge_type = record.metadata.get("edge_type")
    if edge_type is not None:
        return cast(tuple[str, str, str], tuple(edge_type))
    graph = record.graph
    if len(graph.edges) == 1:
        return next(iter(graph.edges))
    try:
        return graph._default_edge_type()
    except AttributeError as exc:
        raise ValueError("Temporal event prediction on heterogeneous graphs requires edge_type") from exc


def _batch_homo_graphs(graphs, *, context):
    if len(graphs) == 1:
        return graphs[0], {id(graphs[0]): 0}
    if any(set(graph.nodes) != {"node"} or len(graph.edges) != 1 for graph in graphs):
        raise ValueError(f"{context} batching multiple graphs currently supports homogeneous graphs only")

    first_graph = graphs[0]
    edge_type = first_graph._default_edge_type()
    time_attr = first_graph.schema.time_attr
    node_keys = tuple(first_graph.nodes["node"].data.keys())
    edge_store = first_graph.edges[edge_type]
    edge_keys = tuple(edge_store.data.keys())
    for graph in graphs[1:]:
        if graph._default_edge_type() != edge_type:
            raise ValueError(f"{context} requires matching edge types when batching multiple graphs")
        if graph.schema.time_attr != time_attr:
            raise ValueError(f"{context} requires matching time_attr when batching multiple graphs")
        if tuple(graph.nodes["node"].data.keys()) != node_keys:
            raise ValueError(f"{context} requires matching node feature keys when batching multiple graphs")
        if tuple(graph.edges[graph._default_edge_type()].data.keys()) != edge_keys:
            raise ValueError(f"{context} requires matching edge feature keys when batching multiple graphs")

    graph_offsets = {}
    offset = 0
    edge_indices = []
    node_data = {key: [] for key in node_keys}
    edge_data = {key: [] for key in edge_keys if key != "edge_index"}
    for graph in graphs:
        graph_offsets[id(graph)] = offset
        num_nodes = int(graph.x.size(0))
        edge_index = graph.edge_index
        edge_count = int(edge_index.size(1))
        edge_indices.append(edge_index + offset)
        for key in node_keys:
            value = graph.nodes["node"].data[key]
            if not _node_aligned_value(value, num_nodes):
                raise ValueError(
                    f"{context} cannot batch multi-graph node attribute '{key}' because it is not node-aligned"
                )
            node_data[key].append(value)
        for key, value in graph.edges[graph._default_edge_type()].data.items():
            if key == "edge_index":
                continue
            if not _node_aligned_value(value, edge_count):
                raise ValueError(
                    f"{context} cannot batch multi-graph edge attribute '{key}' because it is not edge-aligned"
                )
            edge_data[key].append(value)
        offset += num_nodes

    batched_node_data = {key: torch.cat(values, dim=0) for key, values in node_data.items()}
    batched_edge_data = {key: torch.cat(values, dim=0) for key, values in edge_data.items()}
    batched_edge_index = torch.cat(edge_indices, dim=1)
    if edge_type == ("node", "to", "node") and time_attr is None:
        batched_graph = Graph.homo(edge_index=batched_edge_index, edge_data=batched_edge_data, **batched_node_data)
    else:
        nodes = {"node": batched_node_data}
        edges = {edge_type: {"edge_index": batched_edge_index, **batched_edge_data}}
        if time_attr is None:
            batched_graph = Graph.hetero(nodes=nodes, edges=edges)
        else:
            batched_graph = Graph.temporal(nodes=nodes, edges=edges, time_attr=time_attr)
    return batched_graph, graph_offsets


def _batch_hetero_graphs(graphs, *, context):
    if len(graphs) == 1:
        graph = graphs[0]
        return graph, {id(graph): {node_type: 0 for node_type in graph.schema.node_types}}

    first_graph = graphs[0]
    node_types = first_graph.schema.node_types
    edge_types = first_graph.schema.edge_types
    time_attr = first_graph.schema.time_attr
    node_keys = {
        node_type: tuple(first_graph.nodes[node_type].data.keys())
        for node_type in node_types
    }
    edge_keys = {
        edge_type: tuple(first_graph.edges[edge_type].data.keys())
        for edge_type in edge_types
    }

    for graph in graphs[1:]:
        if graph.schema.node_types != node_types or graph.schema.edge_types != edge_types:
            raise ValueError(f"{context} requires matching node and edge types when batching multiple graphs")
        if graph.schema.time_attr != time_attr:
            raise ValueError(f"{context} requires matching time_attr when batching multiple graphs")
        for node_type in node_types:
            if tuple(graph.nodes[node_type].data.keys()) != node_keys[node_type]:
                raise ValueError(f"{context} requires matching node feature keys when batching multiple graphs")
        for edge_type in edge_types:
            if tuple(graph.edges[edge_type].data.keys()) != edge_keys[edge_type]:
                raise ValueError(f"{context} requires matching edge feature keys when batching multiple graphs")

    graph_offsets = {}
    running_offsets = {node_type: 0 for node_type in node_types}
    batched_nodes = {
        node_type: {key: [] for key in node_keys[node_type]}
        for node_type in node_types
    }
    batched_edges = {
        edge_type: {key: [] for key in edge_keys[edge_type] if key != "edge_index"}
        for edge_type in edge_types
    }
    edge_indices = {edge_type: [] for edge_type in edge_types}

    for graph in graphs:
        graph_offsets[id(graph)] = dict(running_offsets)
        node_counts = {}
        for node_type in node_types:
            store = graph.nodes[node_type]
            node_count = _node_count(store)
            node_counts[node_type] = node_count
            for key in node_keys[node_type]:
                value = store.data[key]
                if not _node_aligned_value(value, node_count):
                    raise ValueError(
                        f"{context} cannot batch multi-graph node attribute '{node_type}.{key}' because it is not node-aligned"
                    )
                batched_nodes[node_type][key].append(value)

        for edge_type in edge_types:
            src_type, _, dst_type = edge_type
            store = graph.edges[edge_type]
            edge_count = int(store.edge_index.size(1))
            offset = torch.tensor(
                [[running_offsets[src_type]], [running_offsets[dst_type]]],
                dtype=store.edge_index.dtype,
                device=store.edge_index.device,
            )
            edge_indices[edge_type].append(store.edge_index + offset)
            for key in edge_keys[edge_type]:
                if key == "edge_index":
                    continue
                value = store.data[key]
                if not _node_aligned_value(value, edge_count):
                    raise ValueError(
                        f"{context} cannot batch multi-graph edge attribute '{edge_type}.{key}' because it is not edge-aligned"
                    )
                batched_edges[edge_type][key].append(value)

        for node_type, node_count in node_counts.items():
            running_offsets[node_type] += node_count

    nodes = {
        node_type: {
            key: torch.cat(values, dim=0)
            for key, values in data.items()
        }
        for node_type, data in batched_nodes.items()
    }
    edges = {}
    for edge_type in edge_types:
        edge_data = {
            key: torch.cat(values, dim=0)
            for key, values in batched_edges[edge_type].items()
        }
        edge_data["edge_index"] = torch.cat(edge_indices[edge_type], dim=1)
        edges[edge_type] = edge_data
    return Graph.hetero(nodes=nodes, edges=edges, time_attr=time_attr), graph_offsets


def _graph_membership_from_counts(counts: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    count_tensor = torch.tensor(counts, dtype=torch.long)
    graph_index = _expand_interval_values(
        torch.arange(len(counts), dtype=torch.long),
        count_tensor,
        step=0,
    )
    graph_ptr = torch.empty(len(counts) + 1, dtype=torch.long)
    graph_ptr[0] = 0
    if count_tensor.numel() > 0:
        graph_ptr[1:] = torch.cumsum(count_tensor, dim=0)
    return graph_index, graph_ptr


def _graph_label_from_graph(graph: Graph, label_key: str) -> int:
    if set(graph.nodes) == {"node"} and label_key in graph.nodes["node"].data:
        value = graph.nodes["node"].data[label_key]
        return _as_python_int(torch.as_tensor(value).reshape(-1)[0])

    candidates = []
    for node_type, store in graph.nodes.items():
        value = store.data.get(label_key)
        if value is None:
            continue
        value = torch.as_tensor(value)
        if value.numel() == 1:
            candidates.append((node_type, _as_python_int(value.reshape(-1)[0])))
    if len(candidates) == 1:
        return candidates[0][1]
    if not candidates:
        raise ValueError(f"GraphBatch could not find graph label {label_key!r} on the graph")
    candidate_types = ", ".join(node_type for node_type, _ in candidates)
    raise ValueError(
        f"GraphBatch found ambiguous graph label {label_key!r} on multiple node types: {candidate_types}"
    )


@dataclass(slots=True)
class GraphBatch:
    graphs: list[Graph]
    graph_index: torch.Tensor | None
    graph_ptr: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    metadata: list[dict] | None = None
    graph_index_by_type: dict[str, torch.Tensor] | None = None
    graph_ptr_by_type: dict[str, torch.Tensor] | None = None

    @classmethod
    def from_graphs(cls, graphs: list[Graph]) -> "GraphBatch":
        if not graphs:
            raise ValueError("GraphBatch requires at least one graph")
        first_graph = graphs[0]
        first_is_homo = set(first_graph.nodes) == {"node"} and len(first_graph.edges) == 1
        if any(((set(graph.nodes) == {"node"} and len(graph.edges) == 1) != first_is_homo) for graph in graphs[1:]):
            raise ValueError("GraphBatch requires either all homogeneous or all heterogeneous graphs")

        if first_is_homo:
            counts = [graph.x.size(0) for graph in graphs]
            graph_index, graph_ptr = _graph_membership_from_counts(counts)
            return cls(graphs=graphs, graph_index=graph_index, graph_ptr=graph_ptr)

        node_types = first_graph.schema.node_types
        edge_types = first_graph.schema.edge_types
        time_attr = first_graph.schema.time_attr
        for graph in graphs[1:]:
            if graph.schema.node_types != node_types or graph.schema.edge_types != edge_types:
                raise ValueError("GraphBatch requires matching node and edge types for heterogeneous graphs")
            if graph.schema.time_attr != time_attr:
                raise ValueError("GraphBatch requires matching time_attr for heterogeneous graphs")

        graph_index_by_type = {}
        graph_ptr_by_type = {}
        for node_type in node_types:
            counts = [_node_count(graph.nodes[node_type]) for graph in graphs]
            graph_index_by_type[node_type], graph_ptr_by_type[node_type] = _graph_membership_from_counts(counts)
        return cls(
            graphs=graphs,
            graph_index=None,
            graph_ptr=None,
            graph_index_by_type=graph_index_by_type,
            graph_ptr_by_type=graph_ptr_by_type,
        )

    @classmethod
    def from_samples(
        cls,
        samples: list["SampleRecord"],
        *,
        label_key: str | None = None,
        label_source: str | None = None,
    ) -> "GraphBatch":
        graphs = [sample.graph for sample in samples]
        batch = cls.from_graphs(graphs)
        batch.metadata = [sample.metadata for sample in samples]
        if label_source is None:
            return batch
        if label_key is None:
            raise ValueError("GraphBatch.from_samples requires label_key when label_source is set")
        if label_source == "graph":
            batch.labels = torch.tensor([
                _graph_label_from_graph(sample.graph, label_key)
                for sample in samples
            ])
        elif label_source == "metadata":
            batch.labels = torch.tensor([int(sample.metadata[label_key]) for sample in samples])
        else:
            raise ValueError(f"Unsupported label_source: {label_source}")
        return batch

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return GraphBatch(
            graphs=[
                graph.to(device=device, dtype=dtype, non_blocking=non_blocking)
                for graph in self.graphs
            ],
            graph_index=None
            if self.graph_index is None
            else _transfer_tensor(
                self.graph_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            graph_ptr=None
            if self.graph_ptr is None
            else _transfer_tensor(
                self.graph_ptr,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            labels=None
            if self.labels is None
            else _transfer_tensor(
                self.labels,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            metadata=self.metadata,
            graph_index_by_type=None
            if self.graph_index_by_type is None
            else {
                node_type: _transfer_tensor(
                    value,
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                )
                for node_type, value in self.graph_index_by_type.items()
            },
            graph_ptr_by_type=None
            if self.graph_ptr_by_type is None
            else {
                node_type: _transfer_tensor(
                    value,
                    device=device,
                    dtype=dtype,
                    non_blocking=non_blocking,
                )
                for node_type, value in self.graph_ptr_by_type.items()
            },
        )

    def pin_memory(self):
        return GraphBatch(
            graphs=[graph.pin_memory() for graph in self.graphs],
            graph_index=None if self.graph_index is None else pin_tensor(self.graph_index),
            graph_ptr=None if self.graph_ptr is None else pin_tensor(self.graph_ptr),
            labels=None if self.labels is None else pin_tensor(self.labels),
            metadata=self.metadata,
            graph_index_by_type=None
            if self.graph_index_by_type is None
            else {node_type: pin_tensor(value) for node_type, value in self.graph_index_by_type.items()},
            graph_ptr_by_type=None
            if self.graph_ptr_by_type is None
            else {node_type: pin_tensor(value) for node_type, value in self.graph_ptr_by_type.items()},
        )


@dataclass(slots=True)
class NodeBatch:
    graph: Graph
    seed_index: torch.Tensor
    metadata: list[dict] | None = None
    blocks: list[Block | HeteroBlock] | None = None

    @classmethod
    def from_samples(
        cls,
        samples: list["SampleRecord"],
    ) -> "NodeBatch":
        if not samples:
            raise ValueError("NodeBatch requires at least one sample")
        if any(sample.subgraph_seed is None for sample in samples):
            raise ValueError("NodeBatch requires subgraph_seed for every sample")

        graphs = _unique_graphs(samples)
        first_graph = graphs[0]
        if set(first_graph.nodes) == {"node"} and len(first_graph.edges) == 1:
            graph, graph_offsets = _batch_homo_graphs(graphs, context="NodeBatch")
            seed_values = []
            for sample in samples:
                sample_graph = sample.graph
                num_nodes = int(sample_graph.x.size(0))
                seed_index = _require_int(sample.subgraph_seed, field_name="subgraph_seed")
                if seed_index < 0 or seed_index >= num_nodes:
                    raise ValueError("NodeBatch subgraph_seed must fall within the sampled graph node range")
                seed_values.append(seed_index + graph_offsets[id(sample_graph)])
        else:
            graph, graph_offsets = _batch_hetero_graphs(graphs, context="NodeBatch")
            seed_values = []
            for sample in samples:
                sample_graph = sample.graph
                node_type = sample.metadata.get("node_type")
                if node_type is None:
                    raise ValueError("NodeBatch requires metadata['node_type'] for heterogeneous sampled graphs")
                if node_type not in sample_graph.nodes:
                    raise ValueError("NodeBatch metadata['node_type'] must exist in the sampled graph")
                num_nodes = _node_count(sample_graph.nodes[node_type])
                seed_index = _require_int(sample.subgraph_seed, field_name="subgraph_seed")
                if seed_index < 0 or seed_index >= num_nodes:
                    raise ValueError("NodeBatch subgraph_seed must fall within the sampled graph node range")
                seed_values.append(seed_index + graph_offsets[id(sample_graph)][node_type])
        return cls(
            graph=graph,
            seed_index=torch.tensor(seed_values, dtype=torch.long),
            metadata=[sample.metadata for sample in samples],
            blocks=_batch_blocks(samples, context="NodeBatch"),
        )

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return NodeBatch(
            graph=self.graph.to(device=device, dtype=dtype, non_blocking=non_blocking),
            seed_index=_transfer_tensor(
                self.seed_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            metadata=self.metadata,
            blocks=None
            if self.blocks is None
            else [block.to(device=device, dtype=dtype, non_blocking=non_blocking) for block in self.blocks],
        )

    def pin_memory(self):
        return NodeBatch(
            graph=self.graph.pin_memory(),
            seed_index=pin_tensor(self.seed_index),
            metadata=self.metadata,
            blocks=None if self.blocks is None else [block.pin_memory() for block in self.blocks],
        )


@dataclass(slots=True)
class LinkPredictionBatch:
    graph: Graph
    src_index: torch.Tensor
    dst_index: torch.Tensor
    labels: torch.Tensor
    edge_types: tuple[tuple[str, str, str], ...] | None = None
    edge_type_index: torch.Tensor | None = None
    edge_type: tuple[str, str, str] | None = None
    src_node_type: str | None = None
    dst_node_type: str | None = None
    query_index: torch.Tensor | None = None
    filter_mask: torch.Tensor | None = None
    metadata: list[dict] | None = None
    blocks: list[Block | HeteroBlock] | None = None

    @classmethod
    def from_records(
        cls,
        records: list["LinkPredictionRecord"],
    ) -> "LinkPredictionBatch":
        if not records:
            raise ValueError("LinkPredictionBatch requires at least one record")
        graphs = _unique_graphs(records)
        first_graph = graphs[0]
        record_edge_types = [_resolve_link_edge_type(record) for record in records]
        edge_types = tuple(dict.fromkeys(record_edge_types))
        edge_type_to_index = {edge_type: index for index, edge_type in enumerate(edge_types)}
        edge_type_index = torch.tensor([edge_type_to_index[edge_type] for edge_type in record_edge_types], dtype=torch.long)
        if len(edge_types) > 1 and any(record.blocks is not None for record in records):
            all_blocks = [
                block
                for record in records
                if record.blocks is not None
                for block in record.blocks
            ]
            if any(isinstance(block, Block) for block in all_blocks):
                raise ValueError("LinkPredictionBatch blocks require a single edge_type")

        is_homo = set(first_graph.nodes) == {"node"} and len(first_graph.edges) == 1
        if is_homo:
            graph, graph_offsets = _batch_homo_graphs(graphs, context="LinkPredictionBatch")
        else:
            graph, graph_offsets = _batch_hetero_graphs(graphs, context="LinkPredictionBatch")

        if len(edge_types) == 1:
            edge_type = edge_types[0]
            src_node_type = edge_type[0]
            dst_node_type = edge_type[2]
        else:
            edge_type = None
            src_node_type = None
            dst_node_type = None

        src_values = []
        dst_values = []
        for record, current_edge_type in zip(records, record_edge_types):
            record_graph = record.graph
            src_index = _as_python_int(record.src_index)
            dst_index = _as_python_int(record.dst_index)
            if current_edge_type not in record_graph.edges:
                raise ValueError("LinkPredictionBatch record edge_type must exist in the source graph")
            current_src_type, _, current_dst_type = current_edge_type
            if is_homo:
                num_nodes = int(record_graph.x.size(0))
                if src_index < 0 or src_index >= num_nodes or dst_index < 0 or dst_index >= num_nodes:
                    raise ValueError("LinkPredictionBatch indices must fall within the source graph node range")
                offset = graph_offsets[id(record_graph)]
                src_values.append(src_index + offset)
                dst_values.append(dst_index + offset)
            else:
                src_count = _node_count(record_graph.nodes[current_src_type])
                dst_count = _node_count(record_graph.nodes[current_dst_type])
                if src_index < 0 or src_index >= src_count or dst_index < 0 or dst_index >= dst_count:
                    raise ValueError("LinkPredictionBatch indices must fall within the source graph node range")
                src_values.append(src_index + graph_offsets[id(record_graph)][current_src_type])
                dst_values.append(dst_index + graph_offsets[id(record_graph)][current_dst_type])

        src_index = torch.tensor(src_values, dtype=torch.long)
        dst_index = torch.tensor(dst_values, dtype=torch.long)
        labels = torch.tensor([float(record.label) for record in records], dtype=torch.float32)
        query_ids = [_resolve_link_query_id(record) for record in records]
        filter_flags = [
            bool(getattr(record, "filter_ranking", False) or record.metadata.get("filter_ranking", False))
            for record in records
        ]
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("LinkPredictionBatch labels must be binary 0/1")
        supervision_edges_by_type: dict[tuple[str, str, str], set[tuple[int, int]]] = {}
        reverse_supervision_edges: dict[tuple[str, str, str], set[tuple[int, int]]] = {}
        for record, current_edge_type in zip(records, record_edge_types):
            if _as_python_int(record.label) != 1:
                continue
            if not (getattr(record, "exclude_seed_edge", False) or bool(record.metadata.get("exclude_seed_edges", False))):
                continue
            record_graph = record.graph
            current_src_type, _, current_dst_type = current_edge_type
            if is_homo:
                offset = graph_offsets[id(record_graph)]
                src_value = _as_python_int(record.src_index) + offset
                dst_value = _as_python_int(record.dst_index) + offset
            else:
                src_value = _as_python_int(record.src_index) + graph_offsets[id(record_graph)][current_src_type]
                dst_value = _as_python_int(record.dst_index) + graph_offsets[id(record_graph)][current_dst_type]
            supervision_edges_by_type.setdefault(current_edge_type, set()).add((src_value, dst_value))
            reverse_edge_type = _resolve_link_reverse_edge_type(record)
            if reverse_edge_type is not None:
                if is_homo:
                    reverse_src = dst_value
                    reverse_dst = src_value
                else:
                    reverse_src_type, _, reverse_dst_type = reverse_edge_type
                    reverse_src = _as_python_int(record.dst_index) + graph_offsets[id(record_graph)][reverse_src_type]
                    reverse_dst = _as_python_int(record.src_index) + graph_offsets[id(record_graph)][reverse_dst_type]
                reverse_supervision_edges.setdefault(reverse_edge_type, set()).add((reverse_src, reverse_dst))

        batch_graph = graph
        for current_edge_type, supervision_edges in supervision_edges_by_type.items():
            batch_graph = _without_supervision_edges_for_type(batch_graph, current_edge_type, supervision_edges)
        for reverse_edge_type, reverse_edges in reverse_supervision_edges.items():
            batch_graph = _without_supervision_edges_for_type(batch_graph, reverse_edge_type, reverse_edges)
        if all(query_id is None for query_id in query_ids):
            query_index = None
        elif any(query_id is None for query_id in query_ids):
            raise ValueError("LinkPredictionBatch requires query_id for either all or none of the records")
        else:
            query_id_map: dict[Hashable, int] = {}
            query_values = []
            for query_id in query_ids:
                if query_id not in query_id_map:
                    query_id_map[query_id] = len(query_id_map)
                query_values.append(query_id_map[query_id])
            query_index = torch.tensor(query_values, dtype=torch.long)
        filter_mask = torch.tensor(filter_flags, dtype=torch.bool)

        return cls(
            graph=batch_graph,
            src_index=src_index,
            dst_index=dst_index,
            labels=labels,
            edge_types=edge_types,
            edge_type_index=edge_type_index,
            edge_type=edge_type,
            src_node_type=src_node_type,
            dst_node_type=dst_node_type,
            query_index=query_index,
            filter_mask=filter_mask,
            metadata=[record.metadata for record in records],
            blocks=_batch_blocks(records, context="LinkPredictionBatch"),
        )

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return LinkPredictionBatch(
            graph=self.graph.to(device=device, dtype=dtype, non_blocking=non_blocking),
            src_index=_transfer_tensor(
                self.src_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            dst_index=_transfer_tensor(
                self.dst_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            labels=_transfer_tensor(
                self.labels,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            edge_types=self.edge_types,
            edge_type_index=None
            if self.edge_type_index is None
            else _transfer_tensor(
                self.edge_type_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            edge_type=self.edge_type,
            src_node_type=self.src_node_type,
            dst_node_type=self.dst_node_type,
            query_index=None
            if self.query_index is None
            else _transfer_tensor(
                self.query_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            filter_mask=None
            if self.filter_mask is None
            else _transfer_tensor(
                self.filter_mask,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            metadata=self.metadata,
            blocks=None
            if self.blocks is None
            else [block.to(device=device, dtype=dtype, non_blocking=non_blocking) for block in self.blocks],
        )

    def pin_memory(self):
        return LinkPredictionBatch(
            graph=self.graph.pin_memory(),
            src_index=pin_tensor(self.src_index),
            dst_index=pin_tensor(self.dst_index),
            labels=pin_tensor(self.labels),
            edge_types=self.edge_types,
            edge_type_index=None if self.edge_type_index is None else pin_tensor(self.edge_type_index),
            edge_type=self.edge_type,
            src_node_type=self.src_node_type,
            dst_node_type=self.dst_node_type,
            query_index=None if self.query_index is None else pin_tensor(self.query_index),
            filter_mask=None if self.filter_mask is None else pin_tensor(self.filter_mask),
            metadata=self.metadata,
            blocks=None if self.blocks is None else [block.pin_memory() for block in self.blocks],
        )


@dataclass(slots=True)
class TemporalEventBatch:
    graph: Graph
    src_index: torch.Tensor
    dst_index: torch.Tensor
    timestamp: torch.Tensor
    labels: torch.Tensor
    event_features: torch.Tensor | None = None
    edge_types: tuple[tuple[str, str, str], ...] | None = None
    edge_type_index: torch.Tensor | None = None
    edge_type: tuple[str, str, str] | None = None
    src_node_type: str | None = None
    dst_node_type: str | None = None
    metadata: list[dict] | None = None

    @classmethod
    def from_records(
        cls,
        records: list["TemporalEventRecord"],
    ) -> "TemporalEventBatch":
        if not records:
            raise ValueError("TemporalEventBatch requires at least one record")
        if any(record.graph.schema.time_attr is None for record in records):
            raise ValueError("TemporalEventBatch requires a temporal graph with schema.time_attr")
        graphs = _unique_graphs(records)
        first_graph = graphs[0]
        record_edge_types = [_resolve_temporal_edge_type(record) for record in records]
        edge_types = tuple(dict.fromkeys(record_edge_types))
        edge_type_to_index = {edge_type: index for index, edge_type in enumerate(edge_types)}
        edge_type_index = torch.tensor([edge_type_to_index[edge_type] for edge_type in record_edge_types], dtype=torch.long)

        is_homo = set(first_graph.nodes) == {"node"} and len(first_graph.edges) == 1
        if is_homo:
            graph, graph_offsets = _batch_homo_graphs(graphs, context="TemporalEventBatch")
        else:
            graph, graph_offsets = _batch_hetero_graphs(graphs, context="TemporalEventBatch")

        if len(edge_types) == 1:
            edge_type = edge_types[0]
            src_node_type = edge_type[0]
            dst_node_type = edge_type[2]
        else:
            edge_type = None
            src_node_type = None
            dst_node_type = None

        src_values = []
        dst_values = []
        for record, current_edge_type in zip(records, record_edge_types):
            record_graph = record.graph
            src_index = _as_python_int(record.src_index)
            dst_index = _as_python_int(record.dst_index)
            if current_edge_type not in record_graph.edges:
                raise ValueError("TemporalEventBatch record edge_type must exist in the source graph")
            current_src_type, _, current_dst_type = current_edge_type
            if is_homo:
                num_nodes = int(record_graph.x.size(0))
                if src_index < 0 or src_index >= num_nodes or dst_index < 0 or dst_index >= num_nodes:
                    raise ValueError("TemporalEventBatch indices must fall within the source graph node range")
                offset = graph_offsets[id(record_graph)]
                src_values.append(src_index + offset)
                dst_values.append(dst_index + offset)
            else:
                src_count = _node_count(record_graph.nodes[current_src_type])
                dst_count = _node_count(record_graph.nodes[current_dst_type])
                if src_index < 0 or src_index >= src_count or dst_index < 0 or dst_index >= dst_count:
                    raise ValueError("TemporalEventBatch indices must fall within the source graph node range")
                src_values.append(src_index + graph_offsets[id(record_graph)][current_src_type])
                dst_values.append(dst_index + graph_offsets[id(record_graph)][current_dst_type])

        event_features = [record.event_features for record in records]
        if all(feature is None for feature in event_features):
            stacked_event_features = None
        elif any(feature is None for feature in event_features):
            raise ValueError("TemporalEventBatch requires event_features for either all or none of the records")
        else:
            stacked_event_features = torch.stack([torch.as_tensor(feature) for feature in event_features], dim=0)
        return cls(
            graph=graph,
            src_index=torch.tensor(src_values, dtype=torch.long),
            dst_index=torch.tensor(dst_values, dtype=torch.long),
            timestamp=torch.tensor([record.timestamp for record in records]),
            labels=torch.tensor([record.label for record in records]),
            event_features=stacked_event_features,
            edge_types=edge_types,
            edge_type_index=edge_type_index,
            edge_type=edge_type,
            src_node_type=src_node_type,
            dst_node_type=dst_node_type,
            metadata=[record.metadata for record in records],
        )

    def history_graph(self, index: int):
        return self.graph.snapshot(_as_python_int(self.timestamp[index]))

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return TemporalEventBatch(
            graph=self.graph.to(device=device, dtype=dtype, non_blocking=non_blocking),
            src_index=_transfer_tensor(
                self.src_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            dst_index=_transfer_tensor(
                self.dst_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            timestamp=_transfer_tensor(
                self.timestamp,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            labels=_transfer_tensor(
                self.labels,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            event_features=None
            if self.event_features is None
            else _transfer_tensor(
                self.event_features,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            edge_types=self.edge_types,
            edge_type_index=None
            if self.edge_type_index is None
            else _transfer_tensor(
                self.edge_type_index,
                device=device,
                dtype=dtype,
                non_blocking=non_blocking,
            ),
            edge_type=self.edge_type,
            src_node_type=self.src_node_type,
            dst_node_type=self.dst_node_type,
            metadata=self.metadata,
        )

    def pin_memory(self):
        return TemporalEventBatch(
            graph=self.graph.pin_memory(),
            src_index=pin_tensor(self.src_index),
            dst_index=pin_tensor(self.dst_index),
            timestamp=pin_tensor(self.timestamp),
            labels=pin_tensor(self.labels),
            event_features=None if self.event_features is None else pin_tensor(self.event_features),
            edge_types=self.edge_types,
            edge_type_index=None if self.edge_type_index is None else pin_tensor(self.edge_type_index),
            edge_type=self.edge_type,
            src_node_type=self.src_node_type,
            dst_node_type=self.dst_node_type,
            metadata=self.metadata,
        )
