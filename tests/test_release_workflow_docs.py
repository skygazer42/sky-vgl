from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_releasing_docs_map_workflow_jobs_to_artifacts_and_local_commands():
    releasing = (REPO_ROOT / "docs" / "releasing.md").read_text(encoding="utf-8")

    assert "## Workflow map" in releasing
    assert "`ci.yml -> package-check`" in releasing
    assert "`publish.yml -> build`" in releasing
    assert "`interop-smoke.yml -> backend-smoke`" in releasing
    assert "`interop-smoke.yml -> all-artifact-smoke`" in releasing
    assert "`release-dists`" in releasing
    assert "`benchmark-hotpaths`" in releasing
    assert "python -m build" in releasing
    assert "python scripts/release_contract_scan.py --artifact-dir dist" in releasing
    assert "python scripts/release_smoke.py --artifact-dir dist --kind all --interop-backend all" in releasing
