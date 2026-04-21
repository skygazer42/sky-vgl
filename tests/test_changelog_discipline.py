from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_changelog_discipline_is_documented_across_readme_and_migration_guide():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (REPO_ROOT / "docs" / "changelog.md").read_text(encoding="utf-8")
    migration = (REPO_ROOT / "docs" / "migration-guide.md").read_text(encoding="utf-8")

    assert "### API" in changelog
    assert "### Performance" in changelog
    assert "### Interop" in changelog
    assert "### Migration" in changelog

    assert "Compatibility changes and migration notes are tracked in `docs/changelog.md`." in readme
    assert "Performance-impacting changes should summarize benchmark method or artifact changes in `docs/changelog.md`." in readme

    assert "Follow `docs/changelog.md` when compatibility behavior changes." in migration
    assert "Use the `Migration` section for deprecation and removal notices, and the `Interop` section for adapter-specific caveats." in migration
