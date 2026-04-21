from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_graph_naming_story_is_documented_consistently():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    core_concepts = (REPO_ROOT / "docs" / "core-concepts.md").read_text(encoding="utf-8")
    contract = (REPO_ROOT / "docs" / "public-surface-contract.md").read_text(encoding="utf-8")

    expected_batch = "`GraphBatch` is the canonical batched graph container for graph-level training inputs."
    expected_view = "`GraphView` is the canonical read-only graph projection for snapshot/window-style access."
    expected_store = "`NodeStore` and `EdgeStore` are lower-level storage-facing graph internals; prefer `Graph`, `GraphView`, and `GraphBatch` in application code."

    assert expected_batch in readme
    assert expected_view in readme
    assert expected_store in readme

    assert expected_batch in core_concepts
    assert expected_view in core_concepts
    assert expected_store in core_concepts

    assert expected_batch in contract
    assert expected_view in contract
    assert expected_store in contract
