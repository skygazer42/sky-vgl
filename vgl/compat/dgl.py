import torch

from vgl.graph import Block, Graph


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


def _data_device(data) -> torch.device | None:
    for value in data.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return None


def _has_count_tensor(data, *, count: int) -> bool:
    for value in data.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.size(0)) == count:
            return True
    return False


def _node_data_from_dgl(dgl_graph, node_type=None):
    nodes = getattr(dgl_graph, "nodes", None)
    if node_type is None or nodes is None or not hasattr(nodes, "__getitem__"):
        return dict(getattr(dgl_graph, "ndata", {}))
    return _frame_data(nodes[node_type])


def _graph_num_nodes(dgl_graph, node_type=None) -> int:
    num_nodes = getattr(dgl_graph, "num_nodes")
    try:
        if node_type is None:
            return int(num_nodes())
        return int(num_nodes(node_type))
    except TypeError:
        return int(num_nodes())


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


def _is_dgl_block(dgl_graph):
    return bool(getattr(dgl_graph, "is_block", False))


def _block_store_types(src_type: str, dst_type: str) -> tuple[str, str]:
    if src_type == dst_type:
        return f"{src_type}__src", f"{dst_type}__dst"
    return src_type, dst_type


def _block_node_data_from_dgl(dgl_block, node_type, *, side: str):
    nodes = getattr(dgl_block, f"{side}nodes", None)
    if nodes is not None and hasattr(nodes, "__getitem__"):
        return _frame_data(nodes[node_type])
    return dict(getattr(dgl_block, f"{side}data", {}))


def _block_num_nodes(dgl_block, *, node_type, side: str) -> int:
    num_nodes = getattr(dgl_block, f"num_{side}_nodes")
    try:
        return int(num_nodes(node_type))
    except TypeError:
        return int(num_nodes())


def _as_long_tensor(value, *, device=None) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if device is None:
        device = tensor.device
    if tensor.device != device or tensor.dtype != torch.long:
        tensor = tensor.to(device=device, dtype=torch.long)
    return tensor.view(-1)


def _normalized_public_ids(data, *, preferred_key, fallback_key, count: int | None = None, device=None) -> torch.Tensor | None:
    value = data.get(preferred_key)
    if fallback_key is not None:
        if preferred_key in data:
            data.pop(fallback_key, None)
        elif fallback_key in data:
            value = data.pop(fallback_key)
    if value is None:
        if count is None:
            return None
        value = torch.arange(count, device=device, dtype=torch.long)
    value = _as_long_tensor(value, device=device)
    data[preferred_key] = value
    return value


def _graph_node_data_from_dgl(dgl_graph, node_type=None):
    import dgl  # type: ignore[import-not-found]

    node_data = _node_data_from_dgl(dgl_graph, node_type)
    count = _graph_num_nodes(dgl_graph, node_type)
    device = _data_device(node_data)
    public_ids = _normalized_public_ids(
        node_data,
        preferred_key="n_id",
        fallback_key=getattr(dgl, "NID", None),
        device=device,
    )
    if public_ids is None and not _has_count_tensor(node_data, count=count):
        _normalized_public_ids(
            node_data,
            preferred_key="n_id",
            fallback_key=None,
            count=count,
            device=device,
        )
    return node_data


def _graph_edge_data_from_dgl(dgl_graph, edge_type=None):
    import dgl  # type: ignore[import-not-found]

    edge_data = _edge_data_from_dgl(dgl_graph, edge_type)
    _normalized_public_ids(
        edge_data,
        preferred_key="e_id",
        fallback_key=getattr(dgl, "EID", None),
        device=_data_device(edge_data),
    )
    return edge_data


def block_from_dgl(dgl_block):
    import dgl  # type: ignore[import-not-found]

    if not _is_dgl_block(dgl_block):
        raise ValueError("block_from_dgl expects a DGL block")

    canonical_etypes = _canonical_etypes(dgl_block)
    if len(canonical_etypes) != 1:
        raise ValueError("block_from_dgl only supports single-relation DGL blocks")

    edge_type = canonical_etypes[0]
    src_type, _, dst_type = edge_type
    src_store_type, dst_store_type = _block_store_types(src_type, dst_type)
    block_edge_type = (src_store_type, edge_type[1], dst_store_type)

    edge_index = _edge_index_from_dgl(dgl_block, edge_type)
    src_node_data = _block_node_data_from_dgl(dgl_block, src_type, side="src")
    dst_node_data = _block_node_data_from_dgl(dgl_block, dst_type, side="dst")
    edge_data = _edge_data_from_dgl(dgl_block, edge_type)

    src_n_id = _normalized_public_ids(
        src_node_data,
        preferred_key="n_id",
        fallback_key=getattr(dgl, "NID", None),
        count=_block_num_nodes(dgl_block, node_type=src_type, side="src"),
        device=edge_index.device,
    )
    dst_n_id = _normalized_public_ids(
        dst_node_data,
        preferred_key="n_id",
        fallback_key=getattr(dgl, "NID", None),
        count=_block_num_nodes(dgl_block, node_type=dst_type, side="dst"),
        device=edge_index.device,
    )

    edge_data["edge_index"] = edge_index
    _normalized_public_ids(
        edge_data,
        preferred_key="e_id",
        fallback_key=getattr(dgl, "EID", None),
        count=int(edge_index.size(1)),
        device=edge_index.device,
    )

    nodes = {src_store_type: src_node_data, dst_store_type: dst_node_data}
    edges = {block_edge_type: edge_data}
    time_attr = getattr(dgl_block, _VGL_TIME_ATTR, None)
    if time_attr is None or not _has_feature(nodes, edges, time_attr):
        block_graph = Graph.hetero(nodes=nodes, edges=edges)
    else:
        block_graph = Graph.temporal(nodes=nodes, edges=edges, time_attr=time_attr)

    return Block(
        graph=block_graph,
        edge_type=edge_type,
        src_type=src_type,
        dst_type=dst_type,
        src_n_id=src_n_id,
        dst_n_id=dst_n_id,
        src_store_type=src_store_type,
        dst_store_type=dst_store_type,
    )


def block_to_dgl(block):
    import dgl  # type: ignore[import-not-found]

    dgl_block = dgl.create_block(
        {block.edge_type: tuple(block.edge_index)},
        num_src_nodes={block.src_type: int(block.src_n_id.numel())},
        num_dst_nodes={block.dst_type: int(block.dst_n_id.numel())},
    )

    src_frame = dgl_block.srcnodes[block.src_type].data
    dst_frame = dgl_block.dstnodes[block.dst_type].data
    edge_frame = dgl_block.edges[block.edge_type].data
    for key, value in block.srcdata.items():
        src_frame[key] = value
    for key, value in block.dstdata.items():
        dst_frame[key] = value
    for key, value in block.edata.items():
        if key != "edge_index":
            edge_frame[key] = value

    setattr(dgl_block, _VGL_GRAPH_KIND_ATTR, "block")
    if block.graph.schema.time_attr is not None:
        setattr(dgl_block, _VGL_TIME_ATTR, block.graph.schema.time_attr)
    return dgl_block


def from_dgl(dgl_graph):
    if _is_dgl_block(dgl_graph):
        raise ValueError("Graph.from_dgl does not accept DGL blocks; use Block.from_dgl(...) or vgl.compat.block_from_dgl(...)")

    if _is_hetero_dgl_graph(dgl_graph):
        nodes = {
            node_type: _graph_node_data_from_dgl(dgl_graph, node_type)
            for node_type in _ntypes(dgl_graph)
        }
        edges = {}
        for edge_type in _canonical_etypes(dgl_graph):
            edge_data = _graph_edge_data_from_dgl(dgl_graph, edge_type)
            edge_data["edge_index"] = _edge_index_from_dgl(dgl_graph, edge_type)
            edges[edge_type] = edge_data
        time_attr = getattr(dgl_graph, _VGL_TIME_ATTR, None)
        if time_attr is not None and _has_feature(nodes, edges, time_attr):
            return Graph.temporal(nodes=nodes, edges=edges, time_attr=time_attr)
        return Graph.hetero(nodes=nodes, edges=edges)

    return Graph.homo(
        edge_index=_edge_index_from_dgl(dgl_graph),
        edge_data=_graph_edge_data_from_dgl(dgl_graph) or None,
        **_graph_node_data_from_dgl(dgl_graph),
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
