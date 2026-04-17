from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_compatibility_sunset_policy_is_documented_consistently():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    migration = (REPO_ROOT / "docs" / "migration-guide.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")

    expected_policy = "Legacy compatibility namespaces stay supported through the current 0.x line"
    expected_notice = "Breaking removals will be announced in the changelog before they ship."
    expected_preference = "New code should migrate to `vgl.graph`, `vgl.dataloading`, `vgl.engine`, `vgl.tasks`, and `vgl.metrics` now."

    assert expected_policy in readme
    assert expected_notice in readme
    assert expected_preference in readme

    assert expected_policy in migration
    assert expected_notice in migration
    assert expected_preference in migration

    assert "### Migration" in changelog
    assert expected_policy in changelog
    assert expected_notice in changelog
