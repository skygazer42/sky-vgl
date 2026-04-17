from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compat_docs_list_backend_smoke_paths_and_caveats():
    compat_doc = (REPO_ROOT / "docs" / "api" / "compat.md").read_text(encoding="utf-8")

    assert "python scripts/interop_smoke.py --backend pyg" in compat_doc
    assert "python scripts/interop_smoke.py --backend dgl" in compat_doc
    assert "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend pyg" in compat_doc
    assert "python scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend dgl" in compat_doc
    assert "python -m pytest -q tests/compat/test_networkx_adapter.py" in compat_doc

    assert "### DGL caveats" in compat_doc
    assert "### PyG caveats" in compat_doc
    assert "### NetworkX caveats" in compat_doc

    assert "只覆盖图级 round-trip" in compat_doc
    assert "只覆盖 `Graph <-> PyG Data` 的公开 round-trip" in compat_doc
    assert "只面向同构图" in compat_doc
    assert "无向 NetworkX 图不会被静默接受" in compat_doc
