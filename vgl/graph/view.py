from dataclasses import dataclass

import torch

from vgl.graph.schema import GraphSchema
from vgl.graph.stores import EdgeStore, NodeStore


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


@dataclass(slots=True)
class GraphView:
    base: object
    nodes: dict[str, NodeStore]
    edges: dict[tuple[str, str, str], EdgeStore]
    schema: GraphSchema

    def _default_edge_type(self):
        edge_type = ("node", "to", "node")
        if edge_type in self.edges:
            return edge_type
        if len(self.edges) == 1:
            return next(iter(self.edges))
        raise AttributeError("edge_index")

    def _node_count(self, node_type: str) -> int:
        store = self.nodes[node_type]
        for value in store.data.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.size(0))
        graph_store = self.graph_store
        if graph_store is not None:
            return _as_python_int(graph_store.num_nodes(node_type))
        raise ValueError(f"Cannot infer node count for node type {node_type!r}")

    def __getattr__(self, name: str):
        try:
            nodes = object.__getattribute__(self, "nodes")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        if "node" in nodes and name in nodes["node"].data:
            return nodes["node"].data[name]
        raise AttributeError(name)

    @property
    def feature_store(self):
        return getattr(self.base, "feature_store", None)

    @property
    def graph_store(self):
        return getattr(self.base, "graph_store", None)

    @property
    def x(self):
        return self.nodes["node"].x

    @property
    def y(self):
        return self.nodes["node"].y

    @property
    def edge_index(self):
        return self.edges[self._default_edge_type()].edge_index

    @property
    def ndata(self):
        return self.nodes["node"].data

    @property
    def edata(self):
        return self.edges[self._default_edge_type()].data

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return GraphView(
            base=self.base,
            schema=self.schema,
            nodes={
                node_type: store.to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                )
                for node_type, store in self.nodes.items()
            },
            edges={
                edge_type: store.to(
                    device=device, dtype=dtype, non_blocking=non_blocking
                )
                for edge_type, store in self.edges.items()
            },
        )

    def pin_memory(self):
        return GraphView(
            base=self.base,
            schema=self.schema,
            nodes={node_type: store.pin_memory() for node_type, store in self.nodes.items()},
            edges={edge_type: store.pin_memory() for edge_type, store in self.edges.items()},
        )
