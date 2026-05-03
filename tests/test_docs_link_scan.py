import subprocess
import sys
from pathlib import Path

from scripts import docs_link_scan


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = REPO_ROOT / "scripts" / "docs_link_scan.py"


def test_docs_link_scan_lists_non_empty_catalog():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    listed = [line for line in completed.stdout.splitlines() if line.startswith("SCAN ")]
    assert listed


def test_docs_link_scan_public_docs_follow_mkdocs_surface():
    public_docs = {path.relative_to(REPO_ROOT).as_posix() for path in docs_link_scan.DocsContext(REPO_ROOT).public_docs()}

    assert "docs/getting-started/installation.md" in public_docs
    assert "docs/guide/index.md" in public_docs
    assert "docs/plans/2026-03-14-gnn-framework.md" not in public_docs
    assert "docs/public-surface-contract.md" not in public_docs
    assert "docs/releasing.md" not in public_docs


def test_docs_link_scan_supports_yaml_list_not_in_nav(tmp_path: Path):
    repo_root = tmp_path / "repo"
    docs_root = repo_root / "docs"
    (docs_root / "plans").mkdir(parents=True)
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (docs_root / "index.md").write_text("# Index\n", encoding="utf-8")
    (docs_root / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (docs_root / "plans" / "private.md").write_text("# Private\n", encoding="utf-8")
    (repo_root / "mkdocs.yml").write_text(
        "\n".join(
            [
                "site_name: Demo",
                "not_in_nav:",
                "  - plans/*",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    public_docs = {path.relative_to(repo_root).as_posix() for path in docs_link_scan.DocsContext(repo_root).public_docs()}

    assert "README.md" in public_docs
    assert "docs/index.md" in public_docs
    assert "docs/guide.md" in public_docs
    assert "docs/plans/private.md" not in public_docs


def test_docs_link_scan_supports_quoted_yaml_list_not_in_nav(tmp_path: Path):
    repo_root = tmp_path / "repo"
    docs_root = repo_root / "docs"
    (docs_root / "plans").mkdir(parents=True)
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (docs_root / "index.md").write_text("# Index\n", encoding="utf-8")
    (docs_root / "plans" / "private.md").write_text("# Private\n", encoding="utf-8")
    (repo_root / "mkdocs.yml").write_text(
        "\n".join(
            [
                "site_name: Demo",
                "not_in_nav:",
                '  - "plans/*"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    public_docs = {path.relative_to(repo_root).as_posix() for path in docs_link_scan.DocsContext(repo_root).public_docs()}

    assert "docs/plans/private.md" not in public_docs


def test_docs_link_scan_supports_flow_style_not_in_nav(tmp_path: Path):
    repo_root = tmp_path / "repo"
    docs_root = repo_root / "docs"
    (docs_root / "plans").mkdir(parents=True)
    (repo_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (docs_root / "index.md").write_text("# Index\n", encoding="utf-8")
    (docs_root / "guide.md").write_text("# Guide\n", encoding="utf-8")
    (docs_root / "plans" / "private.md").write_text("# Private\n", encoding="utf-8")
    (repo_root / "mkdocs.yml").write_text(
        "\n".join(
            [
                "site_name: Demo",
                'not_in_nav: ["plans/*"]',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    public_docs = {path.relative_to(repo_root).as_posix() for path in docs_link_scan.DocsContext(repo_root).public_docs()}

    assert "docs/index.md" in public_docs
    assert "docs/guide.md" in public_docs
    assert "docs/plans/private.md" not in public_docs


def test_docs_link_scan_lists_nested_public_docs_links():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "docs/getting-started/installation.md link ../support-matrix.md resolves" in completed.stdout


def test_docs_link_scan_passes_on_public_docs_surface():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY" in completed.stdout
    assert "passed" in completed.stdout


def test_docs_link_scan_resolves_local_image_sources(tmp_path: Path):
    repo_root = tmp_path / "repo"
    docs_root = repo_root / "docs"
    assets_root = repo_root / "assets"
    docs_root.mkdir(parents=True)
    assets_root.mkdir(parents=True)
    (assets_root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (repo_root / "README.md").write_text(
        '<p align="center"><img src="assets/logo.png" alt="Demo"/></p>\n',
        encoding="utf-8",
    )
    (docs_root / "index.md").write_text("# Demo\n", encoding="utf-8")

    tasks = docs_link_scan.build_tasks(repo_root)
    image_tasks = [task for task in tasks if task.category == "asset"]

    assert len(image_tasks) == 1
    assert image_tasks[0].description == "README.md asset assets/logo.png resolves"
    assert image_tasks[0].check() == (True, "README.md -> assets/logo.png")
