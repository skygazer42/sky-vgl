from dataclasses import dataclass

import torch

from vgl.graph.graph import Graph


def _transfer_tensor(tensor: torch.Tensor, *, device=None, dtype=None, non_blocking: bool = False) -> torch.Tensor:
    can_cast = dtype is not None and (tensor.is_floating_point() or tensor.is_complex())
    if can_cast:
        return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
    return tensor.to(device=device, non_blocking=non_blocking)


@dataclass(slots=True)
class Block:
    graph: Graph
    edge_type: tuple[str, str, str]
    src_type: str
    dst_type: str
    src_n_id: torch.Tensor
    dst_n_id: torch.Tensor
    src_store_type: str
    dst_store_type: str

    @property
    def block_edge_type(self) -> tuple[str, str, str]:
        return (self.src_store_type, self.edge_type[1], self.dst_store_type)

    @property
    def srcdata(self):
        return self.graph.nodes[self.src_store_type].data

    @property
    def dstdata(self):
        return self.graph.nodes[self.dst_store_type].data

    @property
    def edata(self):
        return self.graph.edges[self.block_edge_type].data

    @property
    def edge_index(self) -> torch.Tensor:
        return self.graph.edges[self.block_edge_type].edge_index

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return Block(
            graph=self.graph.to(device=device, dtype=dtype, non_blocking=non_blocking),
            edge_type=self.edge_type,
            src_type=self.src_type,
            dst_type=self.dst_type,
            src_n_id=_transfer_tensor(self.src_n_id, device=device, dtype=None, non_blocking=non_blocking),
            dst_n_id=_transfer_tensor(self.dst_n_id, device=device, dtype=None, non_blocking=non_blocking),
            src_store_type=self.src_store_type,
            dst_store_type=self.dst_store_type,
        )

    def pin_memory(self):
        return Block(
            graph=self.graph.pin_memory(),
            edge_type=self.edge_type,
            src_type=self.src_type,
            dst_type=self.dst_type,
            src_n_id=self.src_n_id.pin_memory(),
            dst_n_id=self.dst_n_id.pin_memory(),
            src_store_type=self.src_store_type,
            dst_store_type=self.dst_store_type,
        )
