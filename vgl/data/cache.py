import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from vgl.data.catalog import DatasetManifest

CACHE_ENV_VAR = "VGL_DATA_CACHE"


def resolve_cache_dir(root: str | Path | None = None) -> Path:
    if root is not None:
        return Path(root)
    configured = os.getenv(CACHE_ENV_VAR)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "vgl" / "data"


def fingerprint_manifest(manifest: DatasetManifest) -> str:
    return manifest.fingerprint()


@dataclass(slots=True)
class DataCache:
    root: str | Path | None = None
    cache_root: Path = Path()

    def __post_init__(self) -> None:
        self.cache_root = resolve_cache_dir(self.root)

    def manifest_dir(self, manifest: DatasetManifest) -> Path:
        return self.cache_root / manifest.name / fingerprint_manifest(manifest)

    def path_for(self, manifest: DatasetManifest, relative_path: str | Path) -> Path:
        return self.manifest_dir(manifest) / Path(relative_path)

    def get_or_create(
        self,
        manifest: DatasetManifest,
        relative_path: str | Path,
        builder: Callable[[Path], None],
    ) -> tuple[Path, bool]:
        path = self.path_for(manifest, relative_path)
        if path.exists():
            return path, True
        path.parent.mkdir(parents=True, exist_ok=True)
        builder(path)
        return path, False


__all__ = ["CACHE_ENV_VAR", "DataCache", "fingerprint_manifest", "resolve_cache_dir"]
