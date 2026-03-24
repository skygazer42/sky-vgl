import torch

from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.dataloading.dataset import ListDataset
from vgl.graph.graph import Graph


class BuiltinDataset(ListDataset):
    name = "builtin"
    version = "1.0"
    metadata = {}

    def __init__(self, split: str = "train"):
        fixtures = self._fixtures()
        if split not in fixtures:
            raise ValueError(f"unknown split: {split}")
        self.split = split
        self.manifest = self._build_manifest(fixtures)
        super().__init__(fixtures[split])

    @classmethod
    def _fixtures(cls):
        raise NotImplementedError

    @classmethod
    def _build_manifest(cls, fixtures) -> DatasetManifest:
        return DatasetManifest(
            name=cls.name,
            version=cls.version,
            metadata={"builtin": True, **dict(cls.metadata)},
            splits=tuple(
                DatasetSplit(name=split_name, size=len(items))
                for split_name, items in fixtures.items()
            ),
        )


class ToyGraphDataset(BuiltinDataset):
    name = "toy-graph"
    metadata = {"task": "graph-classification", "source": "fixture"}

    @classmethod
    def _fixtures(cls):
        return {
            "train": [
                Graph.homo(
                    edge_index=torch.tensor([[0, 1], [1, 2]]),
                    x=torch.tensor([[1.0], [0.0], [1.0]]),
                    y=torch.tensor([1]),
                ),
                Graph.homo(
                    edge_index=torch.tensor([[0], [1]]),
                    x=torch.tensor([[0.0], [1.0]]),
                    y=torch.tensor([0]),
                ),
            ],
            "test": [
                Graph.homo(
                    edge_index=torch.tensor([[0], [1]]),
                    x=torch.tensor([[1.0], [1.0]]),
                    y=torch.tensor([1]),
                ),
            ],
        }


__all__ = ["BuiltinDataset", "ToyGraphDataset"]
