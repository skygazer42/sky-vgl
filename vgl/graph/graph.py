from dataclasses import dataclass

import torch

from vgl.graph.schema import GraphSchema
from vgl.graph.stores import EdgeStore, NodeStore
from vgl.graph.view import GraphView, _as_python_int


@dataclass(slots=True)
class Graph:
    schema: GraphSchema
    nodes: dict[str, NodeStore]
    edges: dict[tuple[str, str, str], EdgeStore]
    feature_store: object | None = None
    graph_store: object | None = None
    allowed_sparse_formats: tuple[str, ...] = ("coo", "csr", "csc")
    created_sparse_formats: tuple[str, ...] = ("coo",)

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
        graph_store = object.__getattribute__(self, "graph_store")
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

    def _mark_sparse_format_created(self, format_name: str) -> None:
        created = set(self.created_sparse_formats)
        created.add(format_name)
        self.created_sparse_formats = tuple(
            fmt for fmt in self.allowed_sparse_formats if fmt in created
        )

    def _materialize_sparse_layout_all(self, layout) -> None:
        from vgl.sparse import from_edge_index

        cache_key = layout.value
        for edge_type, store in self.edges.items():
            if cache_key in store.adjacency_cache:
                continue
            src_type, _, dst_type = edge_type
            store.adjacency_cache[cache_key] = from_edge_index(
                store.edge_index,
                shape=(self._node_count(src_type), self._node_count(dst_type)),
                layout=layout,
            )
        self._mark_sparse_format_created(cache_key)

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
        return cls(
            schema=schema,
            nodes=node_stores,
            edges=edge_stores,
            feature_store=feature_store,
            graph_store=graph_store,
        )

    @classmethod
    def from_pyg(cls, data):
        from vgl.compat.pyg import from_pyg

        return from_pyg(data)

    @classmethod
    def from_dgl(cls, graph):
        from vgl.compat.dgl import from_dgl

        return from_dgl(graph)

    @classmethod
    def from_networkx(cls, graph):
        from vgl.compat.networkx import from_networkx

        return from_networkx(graph)

    @classmethod
    def from_edge_list(cls, edge_list, *, num_nodes=None, node_data=None, edge_data=None):
        from vgl.compat.edgelist import from_edge_list

        return from_edge_list(edge_list, num_nodes=num_nodes, node_data=node_data, edge_data=edge_data)

    @classmethod
    def from_edge_list_csv(
        cls,
        path,
        *,
        src_column="src",
        dst_column="dst",
        edge_columns=None,
        delimiter=",",
        num_nodes=None,
    ):
        from vgl.compat.edge_list_csv import from_edge_list_csv

        return from_edge_list_csv(
            path,
            src_column=src_column,
            dst_column=dst_column,
            edge_columns=edge_columns,
            delimiter=delimiter,
            num_nodes=num_nodes,
        )

    @classmethod
    def from_csv_tables(
        cls,
        nodes_path,
        edges_path,
        *,
        node_id_column="node_id",
        src_column="src",
        dst_column="dst",
        node_columns=None,
        edge_columns=None,
        delimiter=",",
    ):
        from vgl.compat.csv_tables import from_csv_tables

        return from_csv_tables(
            nodes_path,
            edges_path,
            node_id_column=node_id_column,
            src_column=src_column,
            dst_column=dst_column,
            node_columns=node_columns,
            edge_columns=edge_columns,
            delimiter=delimiter,
        )

    def to_pyg(self):
        from vgl.compat.pyg import to_pyg

        return to_pyg(self)

    def to_dgl(self):
        from vgl.compat.dgl import to_dgl

        return to_dgl(self)

    def to_networkx(self):
        from vgl.compat.networkx import to_networkx

        return to_networkx(self)

    def to_edge_list(self):
        from vgl.compat.edgelist import to_edge_list

        return to_edge_list(self)

    def to_edge_list_csv(
        self,
        path,
        *,
        src_column="src",
        dst_column="dst",
        edge_columns=None,
        delimiter=",",
    ):
        from vgl.compat.edge_list_csv import to_edge_list_csv

        return to_edge_list_csv(
            self,
            path,
            src_column=src_column,
            dst_column=dst_column,
            edge_columns=edge_columns,
            delimiter=delimiter,
        )

    def to_csv_tables(
        self,
        nodes_path,
        edges_path,
        *,
        node_id_column="node_id",
        src_column="src",
        dst_column="dst",
        node_columns=None,
        edge_columns=None,
        delimiter=",",
    ):
        from vgl.compat.csv_tables import to_csv_tables

        return to_csv_tables(
            self,
            nodes_path,
            edges_path,
            node_id_column=node_id_column,
            src_column=src_column,
            dst_column=dst_column,
            node_columns=node_columns,
            edge_columns=edge_columns,
            delimiter=delimiter,
        )

    def adjacency(self, *, layout="coo", edge_type=None):
        from vgl.sparse import SparseLayout, from_edge_index

        if edge_type is None:
            edge_type = self._default_edge_type()
        if isinstance(layout, str):
            layout = SparseLayout(layout.lower())
        if layout.value not in self.allowed_sparse_formats:
            raise ValueError(f"sparse format {layout.value!r} is not enabled for this graph")
        store = self.edges[tuple(edge_type)]
        cache_key = layout.value
        cached = store.adjacency_cache.get(cache_key)
        if cached is not None:
            return cached
        if layout is not SparseLayout.COO:
            self._materialize_sparse_layout_all(layout)
            return store.adjacency_cache[cache_key]
        src_type, _, dst_type = tuple(edge_type)
        adjacency = from_edge_index(
            store.edge_index,
            shape=(self._node_count(src_type), self._node_count(dst_type)),
            layout=layout,
        )
        store.adjacency_cache[cache_key] = adjacency
        self._mark_sparse_format_created(cache_key)
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

    def to_simple(self, *, edge_type=None, count_attr=None):
        from vgl.ops import to_simple

        return to_simple(self, edge_type=edge_type, count_attr=count_attr)

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

    def num_nodes(self, node_type=None):
        from vgl.ops import num_nodes

        return num_nodes(self, node_type)

    def number_of_nodes(self, node_type=None):
        from vgl.ops import number_of_nodes

        return number_of_nodes(self, node_type)

    def num_edges(self, edge_type=None):
        from vgl.ops import num_edges

        return num_edges(self, edge_type)

    def number_of_edges(self, edge_type=None):
        from vgl.ops import number_of_edges

        return number_of_edges(self, edge_type)

    def all_edges(self, *, form: str = "uv", order: str | None = "eid", edge_type=None):
        from vgl.ops import all_edges

        return all_edges(self, form=form, order=order, edge_type=edge_type)

    def formats(self, formats=None):
        from vgl.ops import formats as formats_op

        return formats_op(self, formats)

    def create_formats_(self):
        from vgl.ops import create_formats_

        return create_formats_(self)

    def adj(self, *, edge_type=None, eweight_name: str | None = None, layout="coo"):
        from vgl.ops import adj

        return adj(self, edge_type=edge_type, eweight_name=eweight_name, layout=layout)

    def laplacian(
        self,
        *,
        edge_type=None,
        normalization=None,
        eweight_name: str | None = None,
        layout="coo",
    ):
        from vgl.ops import laplacian

        return laplacian(
            self,
            edge_type=edge_type,
            normalization=normalization,
            eweight_name=eweight_name,
            layout=layout,
        )

    def adj_external(
        self,
        transpose: bool = False,
        *,
        scipy_fmt: str | None = None,
        torch_fmt: str | None = None,
        edge_type=None,
    ):
        from vgl.ops import adj_external

        return adj_external(
            self,
            transpose=transpose,
            scipy_fmt=scipy_fmt,
            torch_fmt=torch_fmt,
            edge_type=edge_type,
        )

    def adj_tensors(self, layout="coo", *, edge_type=None):
        from vgl.ops import adj_tensors

        return adj_tensors(self, layout=layout, edge_type=edge_type)

    def inc(self, typestr: str = "both", *, layout="coo", edge_type=None):
        from vgl.ops import inc

        return inc(self, typestr=typestr, layout=layout, edge_type=edge_type)

    def in_subgraph(self, nodes):
        from vgl.ops import in_subgraph

        return in_subgraph(self, nodes)

    def out_subgraph(self, nodes):
        from vgl.ops import out_subgraph

        return out_subgraph(self, nodes)

    def in_edges(self, v, *, form: str = "uv", edge_type=None):
        from vgl.ops import in_edges

        return in_edges(self, v, form=form, edge_type=edge_type)

    def in_degrees(self, v=None, *, edge_type=None):
        from vgl.ops import in_degrees

        return in_degrees(self, v, edge_type=edge_type)

    def out_edges(self, u, *, form: str = "uv", edge_type=None):
        from vgl.ops import out_edges

        return out_edges(self, u, form=form, edge_type=edge_type)

    def out_degrees(self, u=None, *, edge_type=None):
        from vgl.ops import out_degrees

        return out_degrees(self, u, edge_type=edge_type)

    def predecessors(self, v, *, edge_type=None):
        from vgl.ops import predecessors

        return predecessors(self, v, edge_type=edge_type)

    def successors(self, v, *, edge_type=None):
        from vgl.ops import successors

        return successors(self, v, edge_type=edge_type)

    def find_edges(self, eids, *, edge_type=None):
        from vgl.ops import find_edges

        return find_edges(self, eids, edge_type=edge_type)

    def edge_ids(self, u, v, *, return_uv: bool = False, edge_type=None):
        from vgl.ops import edge_ids

        return edge_ids(self, u, v, return_uv=return_uv, edge_type=edge_type)

    def has_edges_between(self, u, v, *, edge_type=None):
        from vgl.ops import has_edges_between

        return has_edges_between(self, u, v, edge_type=edge_type)

    def reverse(self, *, copy_ndata: bool = True, copy_edata: bool = False):
        from vgl.ops import reverse

        return reverse(self, copy_ndata=copy_ndata, copy_edata=copy_edata)

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

    def to_hetero_block(self, dst_nodes_by_type, *, edge_types=None, include_dst_in_src: bool = True):
        from vgl.ops import to_hetero_block

        return to_hetero_block(
            self,
            dst_nodes_by_type,
            edge_types=edge_types,
            include_dst_in_src=include_dst_in_src,
        )

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
            graph_store=self.graph_store,
            allowed_sparse_formats=self.allowed_sparse_formats,
            created_sparse_formats=self.created_sparse_formats,
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
            graph_store=self.graph_store,
            allowed_sparse_formats=self.allowed_sparse_formats,
            created_sparse_formats=self.created_sparse_formats,
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
