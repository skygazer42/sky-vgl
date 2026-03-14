import torch

from gnn.core.graph import Graph


def from_dgl(dgl_graph):
    src, dst = dgl_graph.edges()
    return Graph.homo(
        edge_index=torch.stack([src, dst]),
        **dict(dgl_graph.ndata),
    )


def to_dgl(graph):
    import dgl

    row, col = graph.edge_index
    dgl_graph = dgl.graph((row, col), num_nodes=graph.x.size(0))
    for key, value in graph.ndata.items():
        dgl_graph.ndata[key] = value
    return dgl_graph
