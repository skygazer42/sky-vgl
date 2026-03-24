from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    name: str
    size: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("split name must be non-empty")
        if int(self.size) < 0:
            raise ValueError("split size must be >= 0")
        object.__setattr__(self, "size", int(self.size))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "size": self.size,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    name: str
    version: str = "0"
    splits: tuple[DatasetSplit, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must be non-empty")
        splits = tuple(self.splits)
        split_names = [split.name for split in splits]
        if len(split_names) != len(set(split_names)):
            raise ValueError("dataset split names must be unique")
        object.__setattr__(self, "splits", splits)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def split(self, name: str) -> DatasetSplit:
        for split in self.splits:
            if split.name == name:
                return split
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metadata": dict(self.metadata),
            "splits": [split.to_dict() for split in self.splits],
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(slots=True)
class DatasetCatalog:
    manifests: dict[str, DatasetManifest] = field(default_factory=dict)

    def register(self, manifest: DatasetManifest) -> DatasetManifest:
        existing = self.manifests.get(manifest.name)
        if existing is not None and existing.fingerprint() != manifest.fingerprint():
            raise ValueError(f"dataset manifest already registered: {manifest.name}")
        self.manifests[manifest.name] = manifest
        return manifest

    def get(self, name: str) -> DatasetManifest:
        return self.manifests[name]

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.manifests))


__all__ = ["DatasetCatalog", "DatasetManifest", "DatasetSplit"]
