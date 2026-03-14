from dataclasses import dataclass

from gnn.core.schema import GraphSchema
from gnn.core.stores import EdgeStore, NodeStore
from gnn.core.view import GraphView


@dataclass(slots=True)
class Graph:
    schema: GraphSchema
    nodes: dict[str, NodeStore]
    edges: dict[tuple[str, str, str], EdgeStore]

    def __getattr__(self, name: str):
        try:
            nodes = object.__getattribute__(self, "nodes")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        if "node" in nodes and name in nodes["node"].data:
            return nodes["node"].data[name]
        raise AttributeError(name)

    @classmethod
    def homo(cls, *, edge_index, **node_data):
        nodes = {"node": NodeStore("node", dict(node_data))}
        edge_type = ("node", "to", "node")
        edges = {edge_type: EdgeStore(edge_type, {"edge_index": edge_index})}
        schema = GraphSchema(
            node_types=("node",),
            edge_types=(edge_type,),
            node_features={"node": tuple(node_data.keys())},
            edge_features={edge_type: ("edge_index",)},
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
        from gnn.compat.pyg import from_pyg

        return from_pyg(data)

    @classmethod
    def from_dgl(cls, graph):
        from gnn.compat.dgl import from_dgl

        return from_dgl(graph)

    def to_pyg(self):
        from gnn.compat.pyg import to_pyg

        return to_pyg(self)

    def to_dgl(self):
        from gnn.compat.dgl import to_dgl

        return to_dgl(self)

    def snapshot(self, t):
        if self.schema.time_attr is None:
            raise ValueError("snapshot requires a temporal graph")
        edges = {}
        for edge_type, store in self.edges.items():
            mask = store.data[self.schema.time_attr] <= t
            edge_data = dict(store.data)
            edge_data["edge_index"] = store.edge_index[:, mask]
            edge_data[self.schema.time_attr] = store.data[self.schema.time_attr][mask]
            edges[edge_type] = EdgeStore(edge_type, edge_data)
        return GraphView(base=self, nodes=self.nodes, edges=edges, schema=self.schema)

    def window(self, *, start, end):
        if self.schema.time_attr is None:
            raise ValueError("window requires a temporal graph")
        edges = {}
        for edge_type, store in self.edges.items():
            timestamps = store.data[self.schema.time_attr]
            mask = (timestamps >= start) & (timestamps <= end)
            edge_data = dict(store.data)
            edge_data["edge_index"] = store.edge_index[:, mask]
            edge_data[self.schema.time_attr] = timestamps[mask]
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
        return self.edges[("node", "to", "node")].edge_index

    @property
    def ndata(self):
        return self.nodes["node"].data

    @property
    def edata(self):
        return self.edges[("node", "to", "node")].data
