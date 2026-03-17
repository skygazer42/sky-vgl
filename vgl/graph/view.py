from dataclasses import dataclass

from vgl.graph.schema import GraphSchema
from vgl.graph.stores import EdgeStore, NodeStore


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

    def __getattr__(self, name: str):
        try:
            nodes = object.__getattribute__(self, "nodes")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        if "node" in nodes and name in nodes["node"].data:
            return nodes["node"].data[name]
        raise AttributeError(name)

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
