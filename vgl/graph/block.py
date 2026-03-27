from dataclasses import dataclass

import torch

from vgl.graph.graph import Graph


def _transfer_tensor(tensor: torch.Tensor, *, device=None, dtype=None, non_blocking: bool = False) -> torch.Tensor:
    can_cast = dtype is not None and (tensor.is_floating_point() or tensor.is_complex())
    if can_cast:
        return tensor.to(device=device, dtype=dtype, non_blocking=non_blocking)
    return tensor.to(device=device, non_blocking=non_blocking)


def _transfer_tensor_dict(values: dict[str, torch.Tensor], *, device=None, dtype=None, non_blocking: bool = False):
    return {
        key: _transfer_tensor(value, device=device, dtype=None, non_blocking=non_blocking)
        for key, value in values.items()
    }


def _pin_tensor_dict(values: dict[str, torch.Tensor]):
    return {key: value.pin_memory() for key, value in values.items()}


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

    @classmethod
    def from_dgl(cls, dgl_block):
        from vgl.compat.dgl import block_from_dgl

        return block_from_dgl(dgl_block)

    def to_dgl(self):
        from vgl.compat.dgl import block_to_dgl

        return block_to_dgl(self)

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


@dataclass(slots=True)
class HeteroBlock:
    graph: Graph
    edge_types: tuple[tuple[str, str, str], ...]
    src_n_id: dict[str, torch.Tensor]
    dst_n_id: dict[str, torch.Tensor]
    src_store_types: dict[str, str]
    dst_store_types: dict[str, str]

    def block_edge_type(self, edge_type) -> tuple[str, str, str]:
        edge_type = tuple(edge_type)
        src_type, rel_type, dst_type = edge_type
        return (
            self.src_store_types[src_type],
            rel_type,
            self.dst_store_types[dst_type],
        )

    def srcdata(self, node_type: str):
        return self.graph.nodes[self.src_store_types[str(node_type)]].data

    def dstdata(self, node_type: str):
        return self.graph.nodes[self.dst_store_types[str(node_type)]].data

    def edata(self, edge_type):
        return self.graph.edges[self.block_edge_type(edge_type)].data

    def edge_index(self, edge_type) -> torch.Tensor:
        return self.graph.edges[self.block_edge_type(edge_type)].edge_index

    def to(self, device=None, dtype=None, non_blocking: bool = False):
        return HeteroBlock(
            graph=self.graph.to(device=device, dtype=dtype, non_blocking=non_blocking),
            edge_types=self.edge_types,
            src_n_id=_transfer_tensor_dict(
                self.src_n_id,
                device=device,
                dtype=None,
                non_blocking=non_blocking,
            ),
            dst_n_id=_transfer_tensor_dict(
                self.dst_n_id,
                device=device,
                dtype=None,
                non_blocking=non_blocking,
            ),
            src_store_types=dict(self.src_store_types),
            dst_store_types=dict(self.dst_store_types),
        )

    def pin_memory(self):
        return HeteroBlock(
            graph=self.graph.pin_memory(),
            edge_types=self.edge_types,
            src_n_id=_pin_tensor_dict(self.src_n_id),
            dst_n_id=_pin_tensor_dict(self.dst_n_id),
            src_store_types=dict(self.src_store_types),
            dst_store_types=dict(self.dst_store_types),
        )
