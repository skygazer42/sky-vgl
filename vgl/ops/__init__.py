from vgl.ops.compact import compact_nodes as compact_nodes
from vgl.ops.khop import khop_nodes as khop_nodes
from vgl.ops.khop import khop_subgraph as khop_subgraph
from vgl.ops.pipeline import GraphTransform as GraphTransform
from vgl.ops.pipeline import TransformPipeline as TransformPipeline
from vgl.ops.structure import add_self_loops as add_self_loops
from vgl.ops.structure import remove_self_loops as remove_self_loops
from vgl.ops.structure import to_bidirected as to_bidirected
from vgl.ops.subgraph import edge_subgraph as edge_subgraph
from vgl.ops.subgraph import node_subgraph as node_subgraph

__all__ = [
    "GraphTransform",
    "TransformPipeline",
    "add_self_loops",
    "remove_self_loops",
    "to_bidirected",
    "node_subgraph",
    "edge_subgraph",
    "khop_nodes",
    "khop_subgraph",
    "compact_nodes",
]
