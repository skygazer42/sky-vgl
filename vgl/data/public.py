from __future__ import annotations

import hashlib
import pickle
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import torch

from vgl.data.cache import resolve_cache_dir
from vgl.data.catalog import DatasetManifest, DatasetSplit
from vgl.data.ondisk import OnDiskGraphDataset
from vgl.dataloading.dataset import ListDataset
from vgl.graph.graph import Graph

_PLANETOID_DATASET_NAMES = {
    "cora": "Cora",
    "citeseer": "Citeseer",
    "pubmed": "PubMed",
}

_PLANETOID_SOURCE_COMMIT = "af6abf468a772cc055bec88c52e4d6f51b7c37e3"
_PLANETOID_RAW_SHA256 = {
    "ind.cora.x": "23cfa55d91c6f624233f5eb7b6e3f141f1bd2b2ae39608a63cf5d084bb27baab",
    "ind.cora.tx": "b9afbaa4a400df991f6f02ef677e1e44da55ffc04801fd02f9e673987829226a",
    "ind.cora.allx": "9419ba2f26f5c35243db64aba110e0f35a04851609e0dc5433450676ca6b8543",
    "ind.cora.y": "94465c14eb53e04ca262198dcbb2521ee8af60fb3f3d546cd6ca24a511b0e7d1",
    "ind.cora.ty": "41f5ac76596a1699cc33f53084a2419e92961c42bb75f36fc38e616d348532ad",
    "ind.cora.ally": "2b998f5cc7fedc86e7f97ca2498f47a1ffc1462c29e2d578b23bf3f62b6e7d71",
    "ind.cora.graph": "58f13302f39dde8852dad6fe6d15b89b077b6d7f837626bce671ef80344b383d",
    "ind.cora.test.index": "297ce89af2b51a6a194181c7dbe1796c6a1cf9cd9349a88383b1b1e227867875",
    "ind.citeseer.x": "d20de19150741e555de0ed037195630fc194114ebc7fd72ced62810148aa22f6",
    "ind.citeseer.tx": "539a8906a2da97628d212f2fdd668c109a8b3f0d36574b59e5d3fd5b5303da21",
    "ind.citeseer.allx": "2ac30345d95c9ec933a817ee0bdd6a5f077f8c184a20e696910192d534414668",
    "ind.citeseer.y": "8ba87e515d8e3ee3cf52f6d2c5b86aa73f11ecad825a808455209ec26cd54924",
    "ind.citeseer.ty": "55e5bd1ba1e733a04598753ad1a8d57957f4b8f89e74519f41f4f4938767e4ce",
    "ind.citeseer.ally": "f704b2d986dde6c2669934de1f3ae5696a6cf9455c0f6b2646a2d135ea0a1c95",
    "ind.citeseer.graph": "d79a4ef9d3e7169aee8946145f6b7306e35bc79bffad26346cfb94e10e9912a7",
    "ind.citeseer.test.index": "2af990671580b6b2df5d158d1038f8292e30c9cd821b5453f399659b25b723d4",
    "ind.pubmed.x": "fbe9cb5c47200d1b7769a26e579be3b7130b556831f9d850e77c4c74f2599b33",
    "ind.pubmed.tx": "4eb5f5d0f30f26497eb0ac6ac9719d09c67e284f1ed0385c93f3ef27a671da3d",
    "ind.pubmed.allx": "508e538e2abdccdd54c4d3e9f49d638b4af7107d85845f66ddea5a5152658b7e",
    "ind.pubmed.y": "e6c807633307a07ed659249006536a147c7999c797f11a2560752990181b41b3",
    "ind.pubmed.ty": "ee2e0d4819d9fbcc403689c3bf2e49face401b605932f093bc58ab6461130fdd",
    "ind.pubmed.ally": "0c37bbe3b5014ec365e9d393cf4791925e5ed94fb194305d74f064a8d7b50f0e",
    "ind.pubmed.graph": "5b89f0036ca22909471f1e6558eb42fa06e0c730a577bde5b058b40638f81dc9",
    "ind.pubmed.test.index": "b101d421688bbaecbd82e8a18f1e282378d6830f3a37666f28ebc47f844341b5",
}

_COMMON_TU_DATASET_NAMES = (
    "MUTAG",
    "PROTEINS",
    "ENZYMES",
    "NCI1",
    "NCI109",
    "PTC_MR",
    "DD",
    "COLLAB",
    "IMDB-BINARY",
    "IMDB-MULTI",
    "REDDIT-BINARY",
    "REDDIT-MULTI-5K",
)

_TU_DATASET_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def _dense_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "toarray"):
        value = value.toarray()
    return torch.as_tensor(value)


def _sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_verified_file(url: str, destination: Path, *, expected_sha256: str) -> None:
    urllib.request.urlretrieve(url, destination)
    digest = _sha256_digest(destination)
    if digest != expected_sha256:
        destination.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {destination.name}")


def _read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def _read_int_lines(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_float_table(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.replace(",", " ").split()]
        rows.append([float(part) for part in parts])
    return rows


def _one_hot_labels(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 1:
        return labels.to(dtype=torch.long)
    return labels.argmax(dim=-1).to(dtype=torch.long)


def _safe_zip_destination(root: Path, member_name: str) -> Path:
    normalized = PurePosixPath(member_name.replace("\\", "/"))
    if normalized.is_absolute() or any(part == ".." for part in normalized.parts):
        raise ValueError(f"unsafe archive member {member_name!r}")
    relative_parts = [part for part in normalized.parts if part not in {"", "."}]
    destination = (root.resolve() / Path(*relative_parts)).resolve()
    destination.relative_to(root.resolve())
    return destination


def _extract_zip_safely(archive: zipfile.ZipFile, destination_dir: Path) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for member in archive.infolist():
        destination = _safe_zip_destination(destination_dir, member.filename)
        if member.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)


class CachedGraphDataset(ListDataset):
    name = "dataset"
    version = "1.0"
    family = "public"
    raw_file_names: tuple[str, ...] = ()
    raw_urls: dict[str, str] = {}

    def __init__(self, root=None, *, transform=None, download: bool = True, force_reload: bool = False):
        self.base_root = resolve_cache_dir(root)
        self.root = Path(self.base_root) / self.dataset_dirname()
        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.transform = transform
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if force_reload:
            for path in self.processed_dir.glob("**/*"):
                if path.is_file():
                    path.unlink()

        if not self.has_processed():
            if not self.has_raw():
                if not download:
                    raise FileNotFoundError(f"{self.__class__.__name__} requires raw files under {self.raw_dir}")
                self.download()
            self.validate_raw()
            self.process()
        else:
            self.validate_processed_cache()

        dataset = OnDiskGraphDataset(self.processed_dir)
        graphs = [dataset[index] for index in range(len(dataset))]
        if transform is not None:
            graphs = [transform(graph) for graph in graphs]
        self.manifest = dataset.manifest
        super().__init__(graphs)

    def dataset_dirname(self) -> str:
        return self.name

    def has_raw(self) -> bool:
        return all((self.raw_dir / filename).exists() for filename in self.raw_file_names)

    def has_processed(self) -> bool:
        return (self.processed_dir / "manifest.json").exists()

    def validate_raw(self) -> None:
        return None

    def validate_processed_cache(self) -> None:
        return None

    def download(self) -> None:
        missing = []
        for filename in self.raw_file_names:
            destination = self.raw_dir / filename
            if destination.exists():
                continue
            url = self.raw_urls.get(filename)
            if url is None:
                missing.append(filename)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, destination)
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"missing raw files for {self.name}: {joined}")

    def build_graphs(self) -> list[Graph]:
        raise NotImplementedError

    def build_manifest(self, graphs: list[Graph]) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            version=self.version,
            metadata={"family": self.family},
            splits=(DatasetSplit("full", size=len(graphs)),),
        )

    def process(self) -> None:
        graphs = self.build_graphs()
        manifest = self.build_manifest(graphs)
        OnDiskGraphDataset.write(self.processed_dir, manifest, graphs)


class KarateClubDataset(CachedGraphDataset):
    name = "karate-club"
    family = "builtin-public"

    def build_graphs(self) -> list[Graph]:
        edge_pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 10), (0, 11),
            (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
            (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
            (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
            (3, 7), (3, 12), (3, 13),
            (4, 6), (4, 10),
            (5, 6), (5, 10), (5, 16),
            (6, 16),
            (8, 30), (8, 32), (8, 33),
            (9, 33),
            (13, 33),
            (14, 32), (14, 33),
            (15, 32), (15, 33),
            (18, 32), (18, 33),
            (19, 33),
            (20, 32), (20, 33),
            (22, 32), (22, 33),
            (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
            (24, 25), (24, 27), (24, 31),
            (25, 31),
            (26, 29), (26, 33),
            (27, 33),
            (28, 31), (28, 33),
            (29, 32), (29, 33),
            (30, 32), (30, 33),
            (31, 32), (31, 33),
            (32, 33),
        ]
        directed = edge_pairs + [(dst, src) for src, dst in edge_pairs]
        edge_index = torch.tensor(directed, dtype=torch.long).t().contiguous()
        x = torch.eye(34, dtype=torch.float32)
        y = torch.tensor([0] * 17 + [1] * 17, dtype=torch.long)
        graph = Graph.homo(edge_index=edge_index, x=x, y=y)
        return [graph]


class PlanetoidDataset(CachedGraphDataset):
    family = "planetoid"

    def __init__(self, root=None, *, name: str, transform=None, download: bool = True, force_reload: bool = False):
        self.dataset_name = str(name)
        normalized = self.dataset_name.lower()
        canonical = _PLANETOID_DATASET_NAMES.get(normalized)
        if canonical is None:
            raise ValueError(f"unsupported Planetoid dataset: {name!r}")
        self.dataset_name = canonical
        self.name = normalized
        prefix = f"ind.{normalized}"
        self.raw_file_names = (
            f"{prefix}.x",
            f"{prefix}.tx",
            f"{prefix}.allx",
            f"{prefix}.y",
            f"{prefix}.ty",
            f"{prefix}.ally",
            f"{prefix}.graph",
            f"{prefix}.test.index",
        )
        self.raw_urls = {
            filename: f"https://raw.githubusercontent.com/kimiyoung/planetoid/{_PLANETOID_SOURCE_COMMIT}/data/{filename}"
            for filename in self.raw_file_names
        }
        super().__init__(root, transform=transform, download=download, force_reload=force_reload)

    def dataset_dirname(self) -> str:
        return f"planetoid/{self.name}"

    def _validate_raw_cache(self) -> None:
        missing = []
        for filename in self.raw_file_names:
            raw_path = self.raw_dir / filename
            expected_sha256 = _PLANETOID_RAW_SHA256.get(filename)
            if not raw_path.exists() or expected_sha256 is None:
                missing.append(filename)
                continue
            if _sha256_digest(raw_path) != expected_sha256:
                raise ValueError(f"checksum mismatch for {filename}")
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"missing raw files for {self.name}: {joined}")

    def validate_raw(self) -> None:
        self._validate_raw_cache()

    def validate_processed_cache(self) -> None:
        self._validate_raw_cache()

    def download(self) -> None:
        missing = []
        for filename in self.raw_file_names:
            destination = self.raw_dir / filename
            if destination.exists():
                continue
            url = self.raw_urls.get(filename)
            expected_sha256 = _PLANETOID_RAW_SHA256.get(filename)
            if url is None or expected_sha256 is None:
                missing.append(filename)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            _download_verified_file(url, destination, expected_sha256=expected_sha256)
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"missing raw files for {self.name}: {joined}")

    def build_manifest(self, graphs: list[Graph]) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            version=self.version,
            metadata={"family": self.family, "dataset": self.dataset_name},
            splits=(DatasetSplit("full", size=len(graphs), metadata={"split_source": "original_masks"}),),
        )

    def build_graphs(self) -> list[Graph]:
        prefix = f"ind.{self.name}"
        tx = _dense_tensor(_read_pickle(self.raw_dir / f"{prefix}.tx")).to(dtype=torch.float32)
        allx = _dense_tensor(_read_pickle(self.raw_dir / f"{prefix}.allx")).to(dtype=torch.float32)
        y = _dense_tensor(_read_pickle(self.raw_dir / f"{prefix}.y"))
        ty = _dense_tensor(_read_pickle(self.raw_dir / f"{prefix}.ty"))
        ally = _dense_tensor(_read_pickle(self.raw_dir / f"{prefix}.ally"))
        graph_dict = _read_pickle(self.raw_dir / f"{prefix}.graph")
        test_index = _read_int_lines(self.raw_dir / f"{prefix}.test.index")

        sorted_test_index = sorted(test_index)
        num_nodes = max(int(allx.size(0)), max(sorted_test_index) + 1 if sorted_test_index else 0)
        num_features = max(int(allx.size(1)), int(tx.size(1)) if tx.ndim > 1 else 1)
        features = torch.zeros((num_nodes, num_features), dtype=torch.float32)
        features[: allx.size(0), : allx.size(1)] = allx
        if sorted_test_index:
            features[torch.tensor(sorted_test_index, dtype=torch.long), : tx.size(1)] = tx

        labels = torch.full((num_nodes,), -1, dtype=torch.long)
        ally_labels = _one_hot_labels(ally)
        labels[: ally_labels.size(0)] = ally_labels
        ty_labels = _one_hot_labels(ty)
        if sorted_test_index:
            labels[torch.tensor(sorted_test_index, dtype=torch.long)] = ty_labels

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[: int(y.size(0))] = True
        val_start = int(y.size(0))
        val_stop = int(allx.size(0))
        if val_stop > val_start:
            val_mask[val_start:val_stop] = True
        if test_index:
            test_mask[torch.tensor(test_index, dtype=torch.long)] = True

        edges = []
        for src_index, neighbors in graph_dict.items():
            for dst_index in neighbors:
                edges.append((int(src_index), int(dst_index)))
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        graph = Graph.homo(
            edge_index=edge_index,
            x=features,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        return [graph]


class TUDataset(CachedGraphDataset):
    family = "tu"

    def __init__(self, root=None, *, name: str, transform=None, download: bool = True, force_reload: bool = False):
        normalized_name = str(name).strip()
        if not normalized_name:
            raise ValueError("TUDataset requires a non-empty dataset name")
        if _TU_DATASET_NAME_RE.fullmatch(normalized_name) is None:
            raise ValueError(f"invalid TUDataset dataset name: {name!r}")
        self.dataset_name = normalized_name.upper()
        self.name = self.dataset_name.lower()
        self.raw_file_names = (
            f"{self.dataset_name}_A.txt",
            f"{self.dataset_name}_graph_indicator.txt",
            f"{self.dataset_name}_graph_labels.txt",
        )
        self.raw_urls = {
            "__archive__": f"https://www.chrsmrrs.com/graphkerneldatasets/{self.dataset_name}.zip",
        }
        super().__init__(root, transform=transform, download=download, force_reload=force_reload)

    def dataset_dirname(self) -> str:
        return f"tu/{self.name}"

    def has_raw(self) -> bool:
        required = [self.raw_dir / filename for filename in self.raw_file_names]
        return all(path.exists() for path in required)

    def download(self) -> None:
        archive_path = self.raw_dir / f"{self.dataset_name}.zip"
        if not archive_path.exists():
            urllib.request.urlretrieve(self.raw_urls["__archive__"], archive_path)
        with zipfile.ZipFile(archive_path) as archive:
            _extract_zip_safely(archive, self.raw_dir)
        nested_dir = self.raw_dir / self.dataset_name
        if nested_dir.is_dir():
            for path in nested_dir.iterdir():
                target = self.raw_dir / path.name
                if not target.exists():
                    path.replace(target)
        missing = [filename for filename in self.raw_file_names if not (self.raw_dir / filename).exists()]
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(f"missing raw files for {self.name}: {joined}")

    def build_manifest(self, graphs: list[Graph]) -> DatasetManifest:
        return DatasetManifest(
            name=self.name,
            version=self.version,
            metadata={"family": self.family, "dataset": self.dataset_name},
            splits=(DatasetSplit("full", size=len(graphs)),),
        )

    def build_graphs(self) -> list[Graph]:
        indicator = _read_int_lines(self.raw_dir / f"{self.dataset_name}_graph_indicator.txt")
        edge_rows = _read_float_table(self.raw_dir / f"{self.dataset_name}_A.txt")
        graph_labels = _read_int_lines(self.raw_dir / f"{self.dataset_name}_graph_labels.txt")
        node_label_path = self.raw_dir / f"{self.dataset_name}_node_labels.txt"
        node_attr_path = self.raw_dir / f"{self.dataset_name}_node_attributes.txt"
        edge_label_path = self.raw_dir / f"{self.dataset_name}_edge_labels.txt"
        edge_attr_path = self.raw_dir / f"{self.dataset_name}_edge_attributes.txt"
        node_labels = _read_int_lines(node_label_path) if node_label_path.exists() else None
        node_attributes = _read_float_table(node_attr_path) if node_attr_path.exists() else None
        edge_labels = _read_int_lines(edge_label_path) if edge_label_path.exists() else None
        edge_attributes = _read_float_table(edge_attr_path) if edge_attr_path.exists() else None

        if node_labels is not None and len(node_labels) != len(indicator):
            raise ValueError(f"{self.dataset_name}_node_labels.txt must align with {self.dataset_name}_graph_indicator.txt")
        if node_attributes is not None and len(node_attributes) != len(indicator):
            raise ValueError(f"{self.dataset_name}_node_attributes.txt must align with {self.dataset_name}_graph_indicator.txt")
        if edge_labels is not None and len(edge_labels) != len(edge_rows):
            raise ValueError(f"{self.dataset_name}_edge_labels.txt must align with {self.dataset_name}_A.txt")
        if edge_attributes is not None and len(edge_attributes) != len(edge_rows):
            raise ValueError(f"{self.dataset_name}_edge_attributes.txt must align with {self.dataset_name}_A.txt")

        graph_ids = sorted(set(indicator))
        global_to_local = {}
        node_ids_by_graph: dict[int, list[int]] = {graph_id: [] for graph_id in graph_ids}
        for global_node_id, graph_id in enumerate(indicator, start=1):
            node_ids_by_graph[graph_id].append(global_node_id)
        for graph_id, node_ids in node_ids_by_graph.items():
            global_to_local[graph_id] = {global_id: local_id for local_id, global_id in enumerate(node_ids)}

        remapped_labels = {label: index for index, label in enumerate(sorted(set(graph_labels)))}
        graphs = []
        for graph_id in graph_ids:
            node_ids = node_ids_by_graph[graph_id]
            mapping = global_to_local[graph_id]
            local_edges = []
            local_edge_labels = []
            local_edge_attributes = []
            for edge_id, (src_value, dst_value) in enumerate(edge_rows):
                src_index = int(src_value)
                dst_index = int(dst_value)
                if src_index not in mapping or dst_index not in mapping:
                    continue
                local_edges.append((mapping[src_index], mapping[dst_index]))
                if edge_labels is not None:
                    local_edge_labels.append(edge_labels[edge_id])
                if edge_attributes is not None:
                    local_edge_attributes.append(edge_attributes[edge_id])
            if local_edges:
                edge_index = torch.tensor(local_edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_data = {}
            if edge_labels is not None:
                edge_data["edge_label"] = torch.tensor(local_edge_labels, dtype=torch.long)
            if edge_attributes is not None:
                if local_edge_attributes:
                    edge_data["edge_attr"] = torch.tensor(local_edge_attributes, dtype=torch.float32)
                else:
                    width = len(edge_attributes[0]) if edge_attributes else 0
                    edge_data["edge_attr"] = torch.empty((0, width), dtype=torch.float32)
            node_data = {}
            if node_labels is not None:
                node_data["node_label"] = torch.tensor([node_labels[node_id - 1] for node_id in node_ids], dtype=torch.long)

            if node_attributes is not None:
                features = torch.tensor(
                    [node_attributes[node_id - 1] for node_id in node_ids],
                    dtype=torch.float32,
                )
            elif node_labels is not None:
                features = torch.tensor([[float(node_labels[node_id - 1])] for node_id in node_ids], dtype=torch.float32)
            else:
                features = torch.ones((len(node_ids), 1), dtype=torch.float32)

            label_value = remapped_labels[graph_labels[graph_id - 1]]
            graphs.append(
                Graph.homo(
                    edge_index=edge_index,
                    edge_data=edge_data,
                    x=features,
                    y=torch.tensor([label_value], dtype=torch.long),
                    **node_data,
                )
            )
        return graphs


class DatasetRegistry:
    _default_registry: "DatasetRegistry | None" = None

    def __init__(self):
        self._builders: dict[str, object] = {}

    def register(self, name: str, builder) -> None:
        self._builders[str(name).lower()] = builder

    @staticmethod
    def _compact_key(name: str) -> str:
        return str(name).strip().lower().replace(" ", "").replace("-", "").replace("_", "")

    @staticmethod
    def _candidate_keys(name: str) -> tuple[str, ...]:
        raw = str(name).strip().lower()
        for separator in (":", "/", "."):
            if separator in raw:
                prefix, suffix = raw.split(separator, 1)
                prefix_candidates = DatasetRegistry._candidate_keys(prefix)
                suffix_candidates = DatasetRegistry._candidate_keys(suffix)
                combined = []
                for prefix_key in prefix_candidates:
                    for suffix_key in suffix_candidates:
                        candidate = f"{prefix_key}{separator}{suffix_key}"
                        if candidate and candidate not in combined:
                            combined.append(candidate)
                return tuple(combined)
        compact = raw.replace(" ", "-")
        candidates = [
            raw,
            compact,
            compact.replace("_", "-"),
            compact.replace("-", "_"),
            DatasetRegistry._compact_key(raw),
        ]
        unique = []
        for candidate in candidates:
            if candidate and candidate not in unique:
                unique.append(candidate)
        return tuple(unique)

    def _resolve_builder(self, name: str):
        for key in self._candidate_keys(name):
            builder = self._builders.get(key)
            if builder is not None:
                return builder
        compact_candidates = {self._compact_key(candidate) for candidate in self._candidate_keys(name)}
        for registered_key, registered_builder in self._builders.items():
            if self._compact_key(registered_key) in compact_candidates:
                return registered_builder
        return None

    @staticmethod
    def _split_prefixed_name(name: str) -> tuple[str, str] | None:
        raw = str(name).strip()
        for separator in (":", "/", "."):
            if separator not in raw:
                continue
            family, dataset_name = raw.split(separator, 1)
            family = family.strip()
            dataset_name = dataset_name.strip()
            if not family or not dataset_name:
                return None
            return family, dataset_name
        lowered = raw.lower()
        for family in ("planetoid", "tu"):
            for separator in ("_", "-"):
                prefix = f"{family}{separator}"
                if not lowered.startswith(prefix):
                    continue
                dataset_name = raw[len(prefix):].strip()
                if not dataset_name:
                    return None
                return family, dataset_name
        return None

    @classmethod
    def _canonical_planetoid_name(cls, name: str) -> str:
        for candidate in cls._candidate_keys(name):
            canonical = _PLANETOID_DATASET_NAMES.get(candidate)
            if canonical is not None:
                return canonical
        return str(name).strip()

    @classmethod
    def _canonical_tu_name(cls, name: str) -> str:
        for candidate in cls._candidate_keys(name):
            for dataset_name in _COMMON_TU_DATASET_NAMES:
                if candidate in cls._candidate_keys(dataset_name):
                    return dataset_name
        return str(name).strip()

    def create(self, name: str, *args, **kwargs):
        prefixed = self._split_prefixed_name(name)
        if prefixed is not None:
            family_name, dataset_name = prefixed
            family_candidates = set(self._candidate_keys(family_name))
            if "planetoid" in family_candidates:
                return PlanetoidDataset(*args, name=self._canonical_planetoid_name(dataset_name), **kwargs)
            if "tu" in family_candidates:
                return TUDataset(*args, name=self._canonical_tu_name(dataset_name), **kwargs)
        builder = self._resolve_builder(name)
        if builder is None:
            available = ", ".join(sorted(self._builders))
            raise KeyError(f"unknown dataset {name!r}; available datasets: {available}")
        return builder(*args, **kwargs)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    @classmethod
    def default(cls) -> "DatasetRegistry":
        if cls._default_registry is None:
            registry = cls()
            registry.register("toy-graph", lambda *args, **kwargs: __import__("vgl.data.datasets", fromlist=["ToyGraphDataset"]).ToyGraphDataset(*args, **kwargs))
            registry.register("karate-club", KarateClubDataset)
            registry.register("cora", lambda *args, **kwargs: PlanetoidDataset(*args, name="Cora", **kwargs))
            registry.register("citeseer", lambda *args, **kwargs: PlanetoidDataset(*args, name="Citeseer", **kwargs))
            registry.register("pubmed", lambda *args, **kwargs: PlanetoidDataset(*args, name="PubMed", **kwargs))
            for dataset_name in _COMMON_TU_DATASET_NAMES:
                registry.register(
                    dataset_name.lower(),
                    lambda *args, _dataset_name=dataset_name, **kwargs: TUDataset(*args, name=_dataset_name, **kwargs),
                )
            cls._default_registry = registry
        return cls._default_registry


__all__ = [
    "CachedGraphDataset",
    "DatasetRegistry",
    "KarateClubDataset",
    "PlanetoidDataset",
    "TUDataset",
]
