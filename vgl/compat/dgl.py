import torch

from vgl.graph import Graph


_DGL_DEFAULT_NODE_TYPE = "_N"
_DGL_DEFAULT_EDGE_TYPE = (_DGL_DEFAULT_NODE_TYPE, "_E", _DGL_DEFAULT_NODE_TYPE)
_VGL_GRAPH_KIND_ATTR = "vgl_graph_kind"
_VGL_TIME_ATTR = "vgl_time_attr"


def _ntypes(dgl_graph):
    ntypes = getattr(dgl_graph, "ntypes", None)
    if ntypes is None:
        return ()
    return tuple(ntypes)


def _canonical_etypes(dgl_graph):
    canonical_etypes = getattr(dgl_graph, "canonical_etypes", None)
    if canonical_etypes is None:
        return ()
    return tuple(tuple(edge_type) for edge_type in canonical_etypes)


def _is_hetero_dgl_graph(dgl_graph):
    if getattr(dgl_graph, _VGL_GRAPH_KIND_ATTR, None) is not None:
        return True
    if getattr(dgl_graph, _VGL_TIME_ATTR, None) is not None:
        return True
    ntypes = _ntypes(dgl_graph)
    canonical_etypes = _canonical_etypes(dgl_graph)
    if not canonical_etypes:
        return False
    return (
        len(ntypes) != 1
        or len(canonical_etypes) != 1
        or ntypes[0] != _DGL_DEFAULT_NODE_TYPE
        or canonical_etypes[0] != _DGL_DEFAULT_EDGE_TYPE
    )


def _frame_data(frame):
    return dict(getattr(frame, "data", frame))


def _node_data_from_dgl(dgl_graph, node_type=None):
    nodes = getattr(dgl_graph, "nodes", None)
    if node_type is None or nodes is None or not hasattr(nodes, "__getitem__"):
        return dict(getattr(dgl_graph, "ndata", {}))
    return _frame_data(nodes[node_type])


def _edge_pairs_from_dgl(dgl_graph, edge_type=None):
    if edge_type is None:
        return dgl_graph.edges()
    try:
        return dgl_graph.edges(etype=edge_type)
    except TypeError:
        return dgl_graph.edges(edge_type)


def _edge_index_from_dgl(dgl_graph, edge_type=None):
    src, dst = _edge_pairs_from_dgl(dgl_graph, edge_type)
    return torch.stack([src, dst])


def _edge_data_from_dgl(dgl_graph, edge_type=None):
    edges = getattr(dgl_graph, "edges", None)
    if edge_type is None or edges is None or not hasattr(edges, "__getitem__"):
        return dict(getattr(dgl_graph, "edata", {}))
    return _frame_data(edges[edge_type])


def _has_feature(nodes, edges, feature_name):
    return any(feature_name in data for data in nodes.values()) or any(
        feature_name in data for data in edges.values()
    )


def _node_count(graph, node_type):
    try:
        return graph._node_count(node_type)
    except ValueError:
        count = 0
        for edge_type, store in graph.edges.items():
            src_type, _, dst_type = edge_type
            edge_index = store.edge_index
            if edge_index.numel() == 0:
                continue
            if src_type == node_type:
                count = max(count, int(edge_index[0].max().item()) + 1)
            if dst_type == node_type:
                count = max(count, int(edge_index[1].max().item()) + 1)
        return count


def _use_homo_export(graph):
    return (
        graph.schema.time_attr is None
        and graph.schema.node_types == ("node",)
        and graph.schema.edge_types == (("node", "to", "node"),)
    )


def from_dgl(dgl_graph):
    if _is_hetero_dgl_graph(dgl_graph):
        nodes = {
            node_type: _node_data_from_dgl(dgl_graph, node_type)
            for node_type in _ntypes(dgl_graph)
        }
        edges = {}
        for edge_type in _canonical_etypes(dgl_graph):
            edge_data = _edge_data_from_dgl(dgl_graph, edge_type)
            edge_data["edge_index"] = _edge_index_from_dgl(dgl_graph, edge_type)
            edges[edge_type] = edge_data
        time_attr = getattr(dgl_graph, _VGL_TIME_ATTR, None)
        if time_attr is not None and _has_feature(nodes, edges, time_attr):
            return Graph.temporal(nodes=nodes, edges=edges, time_attr=time_attr)
        return Graph.hetero(nodes=nodes, edges=edges)

    return Graph.homo(
        edge_index=_edge_index_from_dgl(dgl_graph),
        edge_data=_edge_data_from_dgl(dgl_graph) or None,
        **_node_data_from_dgl(dgl_graph),
    )


def to_dgl(graph):
    import dgl  # type: ignore[import-not-found]

    if _use_homo_export(graph):
        row, col = graph.edge_index
        dgl_graph = dgl.graph((row, col), num_nodes=_node_count(graph, "node"))
        for key, value in graph.ndata.items():
            dgl_graph.ndata[key] = value
        for key, value in graph.edata.items():
            if key != "edge_index":
                dgl_graph.edata[key] = value
        return dgl_graph

    data_dict = {
        edge_type: tuple(store.edge_index)
        for edge_type, store in graph.edges.items()
    }
    num_nodes_dict = {
        node_type: _node_count(graph, node_type)
        for node_type in graph.schema.node_types
    }
    dgl_graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    for node_type, store in graph.nodes.items():
        for key, value in store.data.items():
            dgl_graph.nodes[node_type].data[key] = value
    for edge_type, store in graph.edges.items():
        for key, value in store.data.items():
            if key != "edge_index":
                dgl_graph.edges[edge_type].data[key] = value
    setattr(dgl_graph, _VGL_GRAPH_KIND_ATTR, "temporal" if graph.schema.time_attr is not None else "hetero")
    if graph.schema.time_attr is not None:
        setattr(dgl_graph, _VGL_TIME_ATTR, graph.schema.time_attr)
    return dgl_graph
