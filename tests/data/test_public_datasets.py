import hashlib
import pickle
import zipfile
from pathlib import Path

import numpy as np
import torch
import pytest

import vgl.data.public as public_data
from vgl import Graph
from vgl.data import DatasetRegistry, KarateClubDataset, PlanetoidDataset, TUDataset


def _write_pickle(path: Path, value) -> None:
    with path.open("wb") as handle:
        pickle.dump(value, handle)


def _sha256_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_planetoid_raw(root: Path, name: str) -> None:
    raw_dir = root / "planetoid" / name.lower() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"ind.{name.lower()}"

    _write_pickle(raw_dir / f"{prefix}.x", np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    _write_pickle(raw_dir / f"{prefix}.y", np.array([[1, 0], [0, 1]], dtype=np.int64))
    _write_pickle(
        raw_dir / f"{prefix}.allx",
        np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float32,
        ),
    )
    _write_pickle(
        raw_dir / f"{prefix}.ally",
        np.array(
            [
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            dtype=np.int64,
        ),
    )
    _write_pickle(raw_dir / f"{prefix}.tx", np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    _write_pickle(raw_dir / f"{prefix}.ty", np.array([[1, 0], [0, 1]], dtype=np.int64))
    _write_pickle(
        raw_dir / f"{prefix}.graph",
        {
            0: [1],
            1: [0, 2],
            2: [1, 3],
            3: [2, 4],
            4: [3, 5],
            5: [4],
        },
    )
    (raw_dir / f"{prefix}.test.index").write_text("4\n5\n", encoding="utf-8")


def _planetoid_hashes(root: Path, name: str) -> dict[str, str]:
    raw_dir = root / "planetoid" / name.lower() / "raw"
    prefix = f"ind.{name.lower()}"
    return {
        filename.name: _sha256_digest(filename)
        for filename in (
            raw_dir / f"{prefix}.x",
            raw_dir / f"{prefix}.tx",
            raw_dir / f"{prefix}.allx",
            raw_dir / f"{prefix}.y",
            raw_dir / f"{prefix}.ty",
            raw_dir / f"{prefix}.ally",
            raw_dir / f"{prefix}.graph",
            raw_dir / f"{prefix}.test.index",
        )
    }


def _patch_planetoid_hashes(monkeypatch, root: Path, name: str) -> None:
    monkeypatch.setattr(
        public_data,
        "_PLANETOID_RAW_SHA256",
        {**public_data._PLANETOID_RAW_SHA256, **_planetoid_hashes(root, name)},
    )


def _write_tu_raw(root: Path, name: str) -> None:
    raw_dir = root / "tu" / name.lower() / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    prefix = raw_dir / name

    (raw_dir / f"{prefix.name}_graph_indicator.txt").write_text("1\n1\n2\n2\n", encoding="utf-8")
    (raw_dir / f"{prefix.name}_A.txt").write_text("1, 2\n2, 1\n3, 4\n4, 3\n", encoding="utf-8")
    (raw_dir / f"{prefix.name}_graph_labels.txt").write_text("-1\n1\n", encoding="utf-8")
    (raw_dir / f"{prefix.name}_node_labels.txt").write_text("1\n2\n1\n1\n", encoding="utf-8")


def _write_tu_raw_with_edge_features(root: Path, name: str) -> None:
    _write_tu_raw(root, name)
    raw_dir = root / "tu" / name.lower() / "raw"
    prefix = raw_dir / name
    (raw_dir / f"{prefix.name}_edge_labels.txt").write_text("1\n2\n3\n4\n", encoding="utf-8")
    (raw_dir / f"{prefix.name}_edge_attributes.txt").write_text(
        "0.1, 1.1\n0.2, 1.2\n0.3, 1.3\n0.4, 1.4\n",
        encoding="utf-8",
    )


def _write_tu_raw_with_node_attributes(root: Path, name: str) -> None:
    _write_tu_raw(root, name)
    raw_dir = root / "tu" / name.lower() / "raw"
    prefix = raw_dir / name
    (raw_dir / f"{prefix.name}_node_attributes.txt").write_text(
        "1.0, 0.0\n0.0, 1.0\n0.5, 0.5\n1.5, 1.0\n",
        encoding="utf-8",
    )


def test_karate_club_dataset_builds_cached_single_graph(tmp_path):
    dataset = KarateClubDataset(root=tmp_path)

    assert len(dataset) == 1
    assert isinstance(dataset[0], Graph)
    assert dataset.manifest.name == "karate-club"
    assert (tmp_path / "karate-club" / "processed" / "manifest.json").exists()

    cached = KarateClubDataset(root=tmp_path)
    assert len(cached) == 1


@pytest.mark.parametrize("name", ["Cora", "Citeseer", "PubMed"])
def test_planetoid_dataset_loads_supported_raw_files_and_masks(tmp_path, monkeypatch, name):
    _write_planetoid_raw(tmp_path, name)
    _patch_planetoid_hashes(monkeypatch, tmp_path, name)

    dataset = PlanetoidDataset(root=tmp_path, name=name, download=False)

    assert len(dataset) == 1
    graph = dataset[0]
    assert isinstance(graph, Graph)
    assert graph.x.size() == (6, 2)
    assert int(graph.train_mask.sum()) == 2
    assert int(graph.val_mask.sum()) == 2
    assert int(graph.test_mask.sum()) == 2
    assert dataset.manifest.metadata["family"] == "planetoid"
    assert dataset.manifest.metadata["dataset"] == name


def test_planetoid_dataset_requires_raw_files_when_download_disabled(tmp_path):
    with pytest.raises(FileNotFoundError, match="raw files"):
        PlanetoidDataset(root=tmp_path, name="Cora", download=False)


def test_planetoid_dataset_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    _write_planetoid_raw(tmp_path, "Cora")
    _patch_planetoid_hashes(monkeypatch, tmp_path, "Cora")
    dataset = PlanetoidDataset(root=tmp_path, name="Cora", download=False)
    missing = dataset.raw_dir / "ind.cora.x"
    missing.unlink()

    def fake_urlretrieve(_url, destination):
        Path(destination).write_bytes(b"corrupted")
        return str(destination), None

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    with pytest.raises(ValueError, match="checksum mismatch"):
        dataset.download()

    assert not missing.exists()


def test_planetoid_dataset_raw_urls_are_pinned_to_an_immutable_commit(tmp_path, monkeypatch):
    _write_planetoid_raw(tmp_path, "Cora")
    _patch_planetoid_hashes(monkeypatch, tmp_path, "Cora")
    dataset = PlanetoidDataset(root=tmp_path, name="Cora", download=False)

    assert all("/master/" not in url for url in dataset.raw_urls.values())


def test_planetoid_dataset_revalidates_cached_raw_hashes_before_deserializing(tmp_path, monkeypatch):
    _write_planetoid_raw(tmp_path, "Cora")
    _patch_planetoid_hashes(monkeypatch, tmp_path, "Cora")
    PlanetoidDataset(root=tmp_path, name="Cora", download=False)
    corrupted = tmp_path / "planetoid" / "cora" / "raw" / "ind.cora.x"
    corrupted.write_bytes(b"corrupted")

    class _FailOnDeserialize:
        def __init__(self, *_args, **_kwargs) -> None:
            raise AssertionError("deserialization should not happen before raw hash revalidation")

    monkeypatch.setattr(public_data, "OnDiskGraphDataset", _FailOnDeserialize)

    with pytest.raises(ValueError, match="checksum mismatch"):
        PlanetoidDataset(root=tmp_path, name="Cora", download=False)


def test_planetoid_dataset_revalidates_raw_hashes_before_first_processing(tmp_path, monkeypatch):
    _write_planetoid_raw(tmp_path, "Cora")
    _patch_planetoid_hashes(monkeypatch, tmp_path, "Cora")
    corrupted = tmp_path / "planetoid" / "cora" / "raw" / "ind.cora.tx"
    corrupted.write_bytes(b"corrupted")

    with pytest.raises(ValueError, match="checksum mismatch"):
        PlanetoidDataset(root=tmp_path, name="Cora", download=False)


@pytest.mark.parametrize("name", ["MUTAG", "PROTEINS", "ENZYMES"])
def test_tu_dataset_loads_standard_tu_raw_files(tmp_path, name):
    _write_tu_raw(tmp_path, name)

    dataset = TUDataset(root=tmp_path, name=name, download=False)

    assert len(dataset) == 2
    assert all(isinstance(dataset[index], Graph) for index in range(len(dataset)))
    assert tuple(int(dataset[index].y.item()) for index in range(len(dataset))) == (0, 1)
    assert dataset.manifest.metadata["family"] == "tu"
    assert dataset.manifest.metadata["dataset"] == name


def test_tu_dataset_preserves_optional_edge_labels_and_attributes(tmp_path):
    _write_tu_raw_with_edge_features(tmp_path, "MUTAG")

    dataset = TUDataset(root=tmp_path, name="MUTAG", download=False)

    first_graph = dataset[0]
    second_graph = dataset[1]
    assert torch.equal(first_graph.edata["edge_label"], torch.tensor([1, 2], dtype=torch.long))
    assert torch.equal(second_graph.edata["edge_label"], torch.tensor([3, 4], dtype=torch.long))
    assert torch.equal(first_graph.edata["edge_attr"], torch.tensor([[0.1, 1.1], [0.2, 1.2]], dtype=torch.float32))
    assert torch.equal(second_graph.edata["edge_attr"], torch.tensor([[0.3, 1.3], [0.4, 1.4]], dtype=torch.float32))


def test_tu_dataset_preserves_node_labels_when_node_attributes_exist(tmp_path):
    _write_tu_raw_with_node_attributes(tmp_path, "MUTAG")

    dataset = TUDataset(root=tmp_path, name="MUTAG", download=False)

    first_graph = dataset[0]
    second_graph = dataset[1]
    assert torch.equal(first_graph.x, torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    assert torch.equal(second_graph.x, torch.tensor([[0.5, 0.5], [1.5, 1.0]], dtype=torch.float32))
    assert torch.equal(first_graph.ndata["node_label"], torch.tensor([1, 2], dtype=torch.long))
    assert torch.equal(second_graph.ndata["node_label"], torch.tensor([1, 1], dtype=torch.long))


@pytest.mark.parametrize("name", ["../mutag", "mutag/../../escape", "mutag\\escape"])
def test_tu_dataset_rejects_unsafe_dataset_names(tmp_path, name):
    with pytest.raises(ValueError, match="dataset name"):
        TUDataset(root=tmp_path, name=name, download=False)

    assert list(tmp_path.rglob("*")) == []


def test_tu_dataset_download_rejects_archive_members_outside_raw_dir(tmp_path):
    _write_tu_raw(tmp_path, "MUTAG")
    dataset = TUDataset(root=tmp_path, name="MUTAG", download=False)
    archive_path = dataset.raw_dir / "MUTAG.zip"
    escape_path = dataset.raw_dir.parent / "escape.txt"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "owned")

    with pytest.raises(ValueError, match="unsafe archive member"):
        dataset.download()

    assert not escape_path.exists()


def test_tu_dataset_download_requires_expected_raw_files(tmp_path):
    _write_tu_raw(tmp_path, "MUTAG")
    dataset = TUDataset(root=tmp_path, name="MUTAG", download=False)
    archive_path = dataset.raw_dir / "MUTAG.zip"
    for path in dataset.raw_dir.glob("MUTAG_*.txt"):
        path.unlink()

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("README.txt", "missing dataset files")

    with pytest.raises(FileNotFoundError, match="missing raw files"):
        dataset.download()


def test_tu_dataset_download_extracts_nested_archive_into_raw_dir(tmp_path):
    _write_tu_raw(tmp_path, "MUTAG")
    dataset = TUDataset(root=tmp_path, name="MUTAG", download=False)
    archive_path = dataset.raw_dir / "MUTAG.zip"
    raw_files = sorted(dataset.raw_dir.glob("MUTAG_*.txt"))

    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in raw_files:
            archive.write(path, arcname=f"MUTAG/{path.name}")

    for path in raw_files:
        path.unlink()

    dataset.download()

    extracted_files = sorted(path.name for path in dataset.raw_dir.glob("MUTAG_*.txt"))
    assert extracted_files == sorted(path.name for path in raw_files)

    reloaded = TUDataset(root=tmp_path, name="MUTAG", download=False)
    assert len(reloaded) == 2


def test_dataset_registry_lists_and_constructs_public_datasets(tmp_path):
    registry = DatasetRegistry.default()

    assert {"toy-graph", "karate-club", "cora", "citeseer", "pubmed", "mutag", "proteins", "enzymes"} <= set(registry.names())
    dataset = registry.create("karate-club", root=tmp_path)
    assert isinstance(dataset, KarateClubDataset)


def test_dataset_registry_constructs_common_tu_aliases(tmp_path):
    _write_tu_raw(tmp_path, "PROTEINS")

    dataset = DatasetRegistry.default().create("proteins", root=tmp_path, download=False)

    assert isinstance(dataset, TUDataset)
    assert dataset.manifest.metadata["dataset"] == "PROTEINS"


def test_dataset_registry_accepts_underscore_aliases(tmp_path):
    _write_tu_raw(tmp_path, "IMDB-BINARY")

    karate = DatasetRegistry.default().create("karate_club", root=tmp_path)
    tu_dataset = DatasetRegistry.default().create("imdb_binary", root=tmp_path, download=False)

    assert isinstance(karate, KarateClubDataset)
    assert isinstance(tu_dataset, TUDataset)
    assert tu_dataset.manifest.metadata["dataset"] == "IMDB-BINARY"


def test_dataset_registry_accepts_compact_aliases(tmp_path):
    _write_tu_raw(tmp_path, "IMDB-BINARY")

    karate = DatasetRegistry.default().create("karateclub", root=tmp_path)
    tu_dataset = DatasetRegistry.default().create("imdbbinary", root=tmp_path, download=False)

    assert isinstance(karate, KarateClubDataset)
    assert isinstance(tu_dataset, TUDataset)
    assert tu_dataset.manifest.metadata["dataset"] == "IMDB-BINARY"


def test_dataset_registry_accepts_family_prefixed_dataset_names(tmp_path, monkeypatch):
    _write_planetoid_raw(tmp_path, "Cora")
    _patch_planetoid_hashes(monkeypatch, tmp_path, "Cora")
    _write_tu_raw(tmp_path, "PROTEINS")

    planetoid = DatasetRegistry.default().create("planetoid:cora", root=tmp_path, download=False)
    tu_dataset = DatasetRegistry.default().create("tu:proteins", root=tmp_path, download=False)

    assert isinstance(planetoid, PlanetoidDataset)
    assert planetoid.manifest.metadata["dataset"] == "Cora"
    assert isinstance(tu_dataset, TUDataset)
    assert tu_dataset.manifest.metadata["dataset"] == "PROTEINS"


def test_dataset_registry_accepts_slash_prefixed_dataset_names(tmp_path, monkeypatch):
    _write_planetoid_raw(tmp_path, "PubMed")
    _patch_planetoid_hashes(monkeypatch, tmp_path, "PubMed")
    _write_tu_raw(tmp_path, "IMDB-BINARY")

    planetoid = DatasetRegistry.default().create("planetoid/pubmed", root=tmp_path, download=False)
    tu_dataset = DatasetRegistry.default().create("tu/imdb_binary", root=tmp_path, download=False)

    assert isinstance(planetoid, PlanetoidDataset)
    assert planetoid.manifest.metadata["dataset"] == "PubMed"
    assert isinstance(tu_dataset, TUDataset)
    assert tu_dataset.manifest.metadata["dataset"] == "IMDB-BINARY"


def test_dataset_registry_accepts_dot_prefixed_dataset_names(tmp_path, monkeypatch):
    _write_planetoid_raw(tmp_path, "Citeseer")
    _patch_planetoid_hashes(monkeypatch, tmp_path, "Citeseer")
    _write_tu_raw(tmp_path, "IMDB-BINARY")

    planetoid = DatasetRegistry.default().create("planetoid.citeseer", root=tmp_path, download=False)
    tu_dataset = DatasetRegistry.default().create("tu.imdb_binary", root=tmp_path, download=False)

    assert isinstance(planetoid, PlanetoidDataset)
    assert planetoid.manifest.metadata["dataset"] == "Citeseer"
    assert isinstance(tu_dataset, TUDataset)
    assert tu_dataset.manifest.metadata["dataset"] == "IMDB-BINARY"


def test_dataset_registry_accepts_underscore_and_hyphen_prefixed_dataset_names(tmp_path, monkeypatch):
    _write_planetoid_raw(tmp_path, "PubMed")
    _patch_planetoid_hashes(monkeypatch, tmp_path, "PubMed")
    _write_tu_raw(tmp_path, "IMDB-BINARY")

    planetoid = DatasetRegistry.default().create("planetoid_pubmed", root=tmp_path, download=False)
    tu_dataset = DatasetRegistry.default().create("tu-imdbbinary", root=tmp_path, download=False)

    assert isinstance(planetoid, PlanetoidDataset)
    assert planetoid.manifest.metadata["dataset"] == "PubMed"
    assert isinstance(tu_dataset, TUDataset)
    assert tu_dataset.manifest.metadata["dataset"] == "IMDB-BINARY"


def test_dataset_registry_accepts_compact_prefixed_dataset_names(tmp_path):
    _write_tu_raw(tmp_path, "IMDB-BINARY")

    tu_dataset = DatasetRegistry.default().create("tu.imdbbinary", root=tmp_path, download=False)

    assert isinstance(tu_dataset, TUDataset)
    assert tu_dataset.manifest.metadata["dataset"] == "IMDB-BINARY"
