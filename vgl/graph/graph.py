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
    feature_store: object | None = None

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
        raise ValueError(f"Cannot infer node count for node type {node_type!r}")

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
    def from_storage(cls, *, schema, feature_store, graph_store):
        node_stores = {
            node_type: NodeStore.from_feature_store(
                node_type,
                schema.node_features.get(node_type, ()),
                feature_store,
            )
            for node_type in schema.node_types
        }
        edge_stores = {
            edge_type: EdgeStore.from_storage(
                edge_type,
                schema.edge_features.get(edge_type, ()),
                feature_store,
                graph_store,
            )
            for edge_type in schema.edge_types
        }
        return cls(schema=schema, nodes=node_stores, edges=edge_stores, feature_store=feature_store)

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

    def adjacency(self, *, layout="coo", edge_type=None):
        from vgl.sparse import SparseLayout, from_edge_index

        if edge_type is None:
            edge_type = self._default_edge_type()
        if isinstance(layout, str):
            layout = SparseLayout(layout.lower())
        store = self.edges[tuple(edge_type)]
        cache_key = layout.value
        cached = store.adjacency_cache.get(cache_key)
        if cached is not None:
            return cached
        src_type, _, dst_type = tuple(edge_type)
        adjacency = from_edge_index(
            store.edge_index,
            shape=(self._node_count(src_type), self._node_count(dst_type)),
            layout=layout,
        )
        store.adjacency_cache[cache_key] = adjacency
        return adjacency

    def add_self_loops(self, *, edge_type=None):
        from vgl.ops import add_self_loops

        return add_self_loops(self, edge_type=edge_type)

    def remove_self_loops(self, *, edge_type=None):
        from vgl.ops import remove_self_loops

        return remove_self_loops(self, edge_type=edge_type)

    def to_bidirected(self, *, edge_type=None):
        from vgl.ops import to_bidirected

        return to_bidirected(self, edge_type=edge_type)

    def line_graph(self, *, edge_type=None, backtracking: bool = True, copy_edata: bool = True):
        from vgl.ops import line_graph

        return line_graph(self, edge_type=edge_type, backtracking=backtracking, copy_edata=copy_edata)

    def metapath_reachable_graph(self, metapath, *, relation_name=None):
        from vgl.ops import metapath_reachable_graph

        return metapath_reachable_graph(self, metapath, relation_name=relation_name)

    def random_walk(self, seeds, *, length: int, edge_type=None):
        from vgl.ops import random_walk

        return random_walk(self, seeds, length=length, edge_type=edge_type)

    def metapath_random_walk(self, seeds, metapath):
        from vgl.ops import metapath_random_walk

        return metapath_random_walk(self, seeds, metapath)

    def node_subgraph(self, node_ids, *, edge_type=None):
        from vgl.ops import node_subgraph

        return node_subgraph(self, node_ids, edge_type=edge_type)

    def edge_subgraph(self, edge_ids, *, edge_type=None):
        from vgl.ops import edge_subgraph

        return edge_subgraph(self, edge_ids, edge_type=edge_type)

    def khop_nodes(self, seeds, *, num_hops: int, direction: str = "out", edge_type=None):
        from vgl.ops import khop_nodes

        return khop_nodes(self, seeds, num_hops=num_hops, direction=direction, edge_type=edge_type)

    def khop_subgraph(self, seeds, *, num_hops: int, direction: str = "out", edge_type=None):
        from vgl.ops import khop_subgraph

        return khop_subgraph(self, seeds, num_hops=num_hops, direction=direction, edge_type=edge_type)

    def compact_nodes(self, node_ids, *, edge_type=None):
        from vgl.ops import compact_nodes

        return compact_nodes(self, node_ids, edge_type=edge_type)

    def to_block(self, dst_nodes, *, edge_type=None, include_dst_in_src: bool = True):
        from vgl.ops import to_block

        return to_block(self, dst_nodes, edge_type=edge_type, include_dst_in_src=include_dst_in_src)

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

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return Graph(
            schema=self.schema,
            feature_store=self.feature_store,
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
        return Graph(
            schema=self.schema,
            feature_store=self.feature_store,
            nodes={node_type: store.pin_memory() for node_type, store in self.nodes.items()},
            edges={edge_type: store.pin_memory() for edge_type, store in self.edges.items()},
        )

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
