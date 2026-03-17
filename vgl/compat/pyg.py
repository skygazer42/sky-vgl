from vgl.graph import Graph


def from_pyg(data):
    node_data = {
        "x": data.x,
    }
    if getattr(data, "y", None) is not None:
        node_data["y"] = data.y
    edge_data = {}
    if getattr(data, "edge_attr", None) is not None:
        edge_data["edge_attr"] = data.edge_attr
    if getattr(data, "edge_weight", None) is not None:
        edge_data["edge_weight"] = data.edge_weight
    return Graph.homo(edge_index=data.edge_index, edge_data=edge_data or None, **node_data)


def to_pyg(graph):
    from torch_geometric.data import Data  # type: ignore[import-not-found]

    kwargs = {
        "x": graph.x,
        "edge_index": graph.edge_index,
    }
    if "y" in graph.nodes["node"].data:
        kwargs["y"] = graph.y
    if "edge_attr" in graph.edata:
        kwargs["edge_attr"] = graph.edata["edge_attr"]
    if "edge_weight" in graph.edata:
        kwargs["edge_weight"] = graph.edata["edge_weight"]
    return Data(**kwargs)
