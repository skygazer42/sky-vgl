import torch

from vgl.data.catalog import DatasetCatalog, DatasetManifest, DatasetSplit


def test_dataset_manifest_exposes_declared_splits_and_stable_fingerprint():
    manifest = DatasetManifest(
        name="toy-graph",
        version="1.0",
        metadata={"task": "graph-classification", "source": "fixture"},
        splits=(
            DatasetSplit("train", size=2),
            DatasetSplit("test", size=1, metadata={"shuffle": False}),
        ),
    )

    assert manifest.split("train").size == 2
    assert manifest.split("test").metadata["shuffle"] is False
    assert manifest.fingerprint() == manifest.fingerprint()

    changed = DatasetManifest(
        name="toy-graph",
        version="1.0",
        metadata={"task": "link-prediction", "source": "fixture"},
        splits=(
            DatasetSplit("train", size=2),
            DatasetSplit("test", size=1, metadata={"shuffle": False}),
        ),
    )

    assert changed.fingerprint() != manifest.fingerprint()


def test_dataset_split_accepts_tensor_size_without_tensor_int(monkeypatch):
    def fail_int(self):
        raise AssertionError("DatasetSplit size should stay off tensor.__int__")

    monkeypatch.setattr(torch.Tensor, "__int__", fail_int)

    split = DatasetSplit("train", size=torch.tensor(2), metadata={"shuffle": False})

    assert split.size == 2
    assert split.metadata["shuffle"] is False


def test_dataset_catalog_registers_and_returns_manifests():
    manifest = DatasetManifest(
        name="toy-graph",
        version="1.0",
        splits=(DatasetSplit("train", size=2),),
    )
    catalog = DatasetCatalog()

    catalog.register(manifest)

    assert catalog.get("toy-graph") is manifest
    assert catalog.names() == ("toy-graph",)
