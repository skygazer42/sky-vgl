import torch

from vgl._optional import import_optional
from vgl.graph import Graph


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _infer_num_nodes_from_edge_index(edge_index: torch.Tensor) -> int:
    if edge_index.numel() == 0:
        return 0
    return _as_python_int(edge_index.to(dtype=torch.long).max()) + 1


def from_pyg(data):
    node_data = {}
    if getattr(data, "x", None) is not None:
        node_data["x"] = data.x
    if getattr(data, "y", None) is not None:
        node_data["y"] = data.y
    if getattr(data, "n_id", None) is not None:
        node_data["n_id"] = data.n_id
    num_nodes = getattr(data, "num_nodes", None)
    if num_nodes is not None:
        resolved_num_nodes = _as_python_int(num_nodes)
        for key, value in node_data.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and int(value.size(0)) != resolved_num_nodes:
                raise ValueError(f"PyG node attribute {key!r} must align with num_nodes")
    if not node_data:
        if num_nodes is None:
            num_nodes = _infer_num_nodes_from_edge_index(data.edge_index)
        node_data["n_id"] = torch.arange(_as_python_int(num_nodes), dtype=torch.long)
    edge_data = {}
    if getattr(data, "edge_attr", None) is not None:
        edge_data["edge_attr"] = data.edge_attr
    if getattr(data, "edge_weight", None) is not None:
        edge_data["edge_weight"] = data.edge_weight
    if getattr(data, "e_id", None) is not None:
        edge_data["e_id"] = data.e_id
    return Graph.homo(edge_index=data.edge_index, edge_data=edge_data or None, **node_data)


def to_pyg(graph):
    Data = import_optional(
        "torch_geometric.data",
        package_name="torch-geometric",
        extra_name="pyg",
        feature_name="PyG interoperability",
    ).Data

    kwargs = {
        "edge_index": graph.edge_index,
        "num_nodes": graph._node_count("node"),
    }
    if "x" in graph.nodes["node"].data:
        kwargs["x"] = graph.x
    if "y" in graph.nodes["node"].data:
        kwargs["y"] = graph.y
    if "n_id" in graph.nodes["node"].data:
        kwargs["n_id"] = graph.n_id
    if "edge_attr" in graph.edata:
        kwargs["edge_attr"] = graph.edata["edge_attr"]
    if "edge_weight" in graph.edata:
        kwargs["edge_weight"] = graph.edata["edge_weight"]
    if "e_id" in graph.edata:
        kwargs["e_id"] = graph.edata["e_id"]
    return Data(**kwargs)
