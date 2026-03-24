from vgl.data.cache import DataCache, fingerprint_manifest, resolve_cache_dir
from vgl.data.catalog import DatasetManifest, DatasetSplit


def _manifest():
    return DatasetManifest(
        name="toy-graph",
        version="1.0",
        splits=(DatasetSplit("train", size=2),),
        metadata={"source": "fixture"},
    )


def test_resolve_cache_dir_prefers_explicit_root(tmp_path, monkeypatch):
    monkeypatch.setenv("VGL_DATA_CACHE", str(tmp_path / "env-cache"))

    resolved = resolve_cache_dir(tmp_path / "explicit-cache")

    assert resolved == tmp_path / "explicit-cache"


def test_data_cache_uses_manifest_fingerprint_for_cache_paths(tmp_path):
    manifest = _manifest()
    cache = DataCache(tmp_path)

    path = cache.path_for(manifest, "graph.pt")

    assert path == tmp_path / manifest.name / fingerprint_manifest(manifest) / "graph.pt"


def test_data_cache_get_or_create_hits_existing_entries(tmp_path):
    manifest = _manifest()
    cache = DataCache(tmp_path)
    calls = []

    def build(path):
        calls.append(path)
        path.write_text("payload")

    first_path, first_hit = cache.get_or_create(manifest, "graph.txt", build)
    second_path, second_hit = cache.get_or_create(manifest, "graph.txt", build)

    assert first_path == second_path
    assert first_hit is False
    assert second_hit is True
    assert len(calls) == 1
    assert second_path.read_text() == "payload"
