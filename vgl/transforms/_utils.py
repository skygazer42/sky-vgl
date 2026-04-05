from __future__ import annotations

from collections.abc import Mapping

import torch

from vgl.graph.graph import Graph

EdgeType = tuple[str, str, str]


def is_homo_graph(graph: Graph) -> bool:
    return set(graph.nodes) == {"node"} and len(graph.edges) == 1 and graph.schema.time_attr is None


def clone_graph(graph: Graph, *, nodes: Mapping[str, dict] | None = None, edges: Mapping[EdgeType, dict] | None = None) -> Graph:
    nodes = {
        node_type: dict(store.data)
        for node_type, store in graph.nodes.items()
    } if nodes is None else {node_type: dict(data) for node_type, data in nodes.items()}
    edges = {
        edge_type: dict(store.data)
        for edge_type, store in graph.edges.items()
    } if edges is None else {edge_type: dict(data) for edge_type, data in edges.items()}

    if is_homo_graph(graph):
        edge_type = graph._default_edge_type()
        cloned = Graph.homo(
            edge_index=edges[edge_type]["edge_index"],
            edge_data={key: value for key, value in edges[edge_type].items() if key != "edge_index"},
            **nodes["node"],
        )
    elif graph.schema.time_attr is not None:
        cloned = Graph.temporal(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr)
    else:
        cloned = Graph.hetero(nodes=nodes, edges=edges, time_attr=graph.schema.time_attr)

    cloned.feature_store = graph.feature_store
    cloned.graph_store = graph.graph_store
    cloned.allowed_sparse_formats = tuple(graph.allowed_sparse_formats)
    cloned.created_sparse_formats = tuple(graph.created_sparse_formats)
    return cloned


def is_node_aligned(value, count: int) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.size(0)) == int(count)


def is_edge_aligned(value, count: int) -> bool:
    return isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.size(0)) == int(count)
