from gnn.core.graph import Graph


def from_pyg(data):
    node_data = {
        "x": data.x,
    }
    if getattr(data, "y", None) is not None:
        node_data["y"] = data.y
    return Graph.homo(edge_index=data.edge_index, **node_data)


def to_pyg(graph):
    from torch_geometric.data import Data

    kwargs = {
        "x": graph.x,
        "edge_index": graph.edge_index,
    }
    if "y" in graph.nodes["node"].data:
        kwargs["y"] = graph.y
    return Data(**kwargs)
