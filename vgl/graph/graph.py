from dataclasses import dataclass

import torch

from vgl.graph.schema import GraphSchema
from vgl.graph.stores import EdgeStore, NodeStore
from vgl.graph.view import GraphView


@dataclass(slots=True)
class Graph:
    schema: GraphSchema
    nodes: dict[str, NodeStore]
    edges: dict[tuple[str, str, str], EdgeStore]

    def _default_edge_type(self):
        edge_type = ("node", "to", "node")
        if edge_type in self.edges:
            return edge_type
        if len(self.edges) == 1:
            return next(iter(self.edges))
        raise AttributeError("edge_index")

    def __getattr__(self, name: str):
        try:
            nodes = object.__getattribute__(self, "nodes")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        if "node" in nodes and name in nodes["node"].data:
            return nodes["node"].data[name]
        raise AttributeError(name)

    @staticmethod
    def _slice_edge_data(store, mask):
        edge_count = int(store.edge_index.size(1))
        edge_data = {}
        for key, value in store.data.items():
            if key == "edge_index":
                edge_data[key] = value[:, mask]
            elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
                edge_data[key] = value[mask]
            else:
                edge_data[key] = value
        return edge_data

    @classmethod
    def homo(cls, *, edge_index, edge_data=None, **node_data):
        nodes = {"node": NodeStore("node", dict(node_data))}
        edge_type = ("node", "to", "node")
        homo_edge_data = {"edge_index": edge_index, **dict(edge_data or {})}
        edges = {edge_type: EdgeStore(edge_type, homo_edge_data)}
        schema = GraphSchema(
            node_types=("node",),
            edge_types=(edge_type,),
            node_features={"node": tuple(node_data.keys())},
            edge_features={edge_type: tuple(homo_edge_data.keys())},
        )
        return cls(schema=schema, nodes=nodes, edges=edges)

    @classmethod
    def hetero(cls, *, nodes, edges, time_attr=None):
        node_stores = {
            node_type: NodeStore(node_type, dict(data))
            for node_type, data in nodes.items()
        }
        edge_stores = {
            edge_type: EdgeStore(edge_type, dict(data))
            for edge_type, data in edges.items()
        }
        schema = GraphSchema(
            node_types=tuple(sorted(node_stores)),
            edge_types=tuple(sorted(edge_stores)),
            node_features={
                node_type: tuple(store.data.keys())
                for node_type, store in node_stores.items()
            },
            edge_features={
                edge_type: tuple(store.data.keys())
                for edge_type, store in edge_stores.items()
            },
            time_attr=time_attr,
        )
        return cls(schema=schema, nodes=node_stores, edges=edge_stores)

    @classmethod
    def temporal(cls, *, nodes, edges, time_attr):
        return cls.hetero(nodes=nodes, edges=edges, time_attr=time_attr)

    @classmethod
    def from_pyg(cls, data):
        from vgl.compat.pyg import from_pyg

        return from_pyg(data)

    @classmethod
    def from_dgl(cls, graph):
        from vgl.compat.dgl import from_dgl

        return from_dgl(graph)

    def to_pyg(self):
        from vgl.compat.pyg import to_pyg

        return to_pyg(self)

    def to_dgl(self):
        from vgl.compat.dgl import to_dgl

        return to_dgl(self)

    def snapshot(self, t):
        if self.schema.time_attr is None:
            raise ValueError("snapshot requires a temporal graph")
        edges = {}
        for edge_type, store in self.edges.items():
            mask = store.data[self.schema.time_attr] <= t
            edge_data = self._slice_edge_data(store, mask)
            edges[edge_type] = EdgeStore(edge_type, edge_data)
        return GraphView(base=self, nodes=self.nodes, edges=edges, schema=self.schema)

    def window(self, *, start, end):
        if self.schema.time_attr is None:
            raise ValueError("window requires a temporal graph")
        edges = {}
        for edge_type, store in self.edges.items():
            timestamps = store.data[self.schema.time_attr]
            mask = (timestamps >= start) & (timestamps <= end)
            edge_data = self._slice_edge_data(store, mask)
            edges[edge_type] = EdgeStore(edge_type, edge_data)
        return GraphView(base=self, nodes=self.nodes, edges=edges, schema=self.schema)

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
