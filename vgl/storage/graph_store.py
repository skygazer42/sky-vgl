from collections.abc import Mapping
from typing import Protocol

import torch

from vgl.sparse import SparseLayout, SparseTensor, from_edge_index

EdgeType = tuple[str, str, str]


class GraphStore(Protocol):
    @property
    def edge_types(self) -> tuple[EdgeType, ...]:
        ...

    def num_nodes(self, node_type: str = "node") -> int:
        ...

    def edge_index(self, edge_type: EdgeType | None = None) -> torch.Tensor:
        ...

    def edge_count(self, edge_type: EdgeType | None = None) -> int:
        ...

    def adjacency(
        self,
        *,
        edge_type: EdgeType | None = None,
        layout: SparseLayout | str = SparseLayout.COO,
    ) -> SparseTensor:
        ...


class InMemoryGraphStore:
    def __init__(
        self,
        edges: Mapping[EdgeType, torch.Tensor],
        *,
        num_nodes: Mapping[str, int],
    ):
        self._edges: dict[EdgeType, torch.Tensor] = {
            edge_type: edge_index for edge_type, edge_index in edges.items()
        }
        self._num_nodes: dict[str, int] = dict(num_nodes)

    @property
    def edge_types(self) -> tuple[EdgeType, ...]:
        return tuple(self._edges)

    def num_nodes(self, node_type: str = "node") -> int:
        return int(self._num_nodes[str(node_type)])

    def _resolve_edge_type(self, edge_type: EdgeType | None) -> EdgeType:
        if edge_type is not None:
            return edge_type
        default_edge_type: EdgeType = ("node", "to", "node")
        if default_edge_type in self._edges:
            return default_edge_type
        if len(self._edges) == 1:
            return next(iter(self._edges))
        raise KeyError("edge_type is required when multiple edge types exist")

    def edge_index(self, edge_type: EdgeType | None = None) -> torch.Tensor:
        resolved = self._resolve_edge_type(edge_type)
        return self._edges[resolved]

    def edge_count(self, edge_type: EdgeType | None = None) -> int:
        return int(self.edge_index(edge_type).size(1))

    def adjacency(
        self,
        *,
        edge_type: EdgeType | None = None,
        layout: SparseLayout | str = SparseLayout.COO,
    ) -> SparseTensor:
        resolved = self._resolve_edge_type(edge_type)
        if isinstance(layout, str):
            layout = SparseLayout(layout.lower())
        src_type, _, dst_type = resolved
        shape = (self._num_nodes[src_type], self._num_nodes[dst_type])
        return from_edge_index(self._edges[resolved], shape=shape, layout=layout)
