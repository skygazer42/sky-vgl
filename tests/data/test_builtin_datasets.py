import pytest

from vgl import Graph
from vgl.data import BuiltinDataset, ToyGraphDataset


def test_toy_graph_dataset_exposes_manifest_and_sequence_api():
    dataset = ToyGraphDataset(split="train")

    assert isinstance(dataset, BuiltinDataset)
    assert len(dataset) == 2
    assert isinstance(dataset[0], Graph)
    assert dataset.manifest.name == "toy-graph"
    assert dataset.manifest.split("train").size == 2
    assert dataset.manifest.split("test").size == 1


def test_toy_graph_dataset_validates_split_names():
    with pytest.raises(ValueError, match="unknown split"):
        ToyGraphDataset(split="validation")
