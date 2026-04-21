import re
from pathlib import Path

from scripts.workflow_contracts import workflow_job_text


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_publish_workflow_requires_testpypi_first_and_single_pypi_trigger():
    workflow_path = REPO_ROOT / ".github" / "workflows" / "publish.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    pypi_token_job = workflow_job_text(workflow_path, "publish-pypi-token")
    pypi_trusted_job = workflow_job_text(workflow_path, "publish-pypi-trusted")

    assert "workflow_dispatch:" in workflow_text
    assert "release:" in workflow_text
    assert "types: [published]" in workflow_text
    assert "push:" not in workflow_text
    assert "startsWith(github.ref, 'refs/tags/v')" not in pypi_token_job
    assert "startsWith(github.ref, 'refs/tags/v')" not in pypi_trusted_job
    assert "github.event_name == 'release'" in pypi_token_job
    assert "github.event_name == 'release'" in pypi_trusted_job


def test_publish_workflow_pins_each_action_to_a_full_commit_sha():
    workflow_text = (REPO_ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

    for action in (
        "actions/checkout",
        "actions/setup-python",
        "actions/upload-artifact",
        "actions/download-artifact",
        "pypa/gh-action-pypi-publish",
    ):
        assert re.search(rf"uses:\s+{re.escape(action)}@[0-9a-f]{{40}}", workflow_text), action

    assert "@v4" not in workflow_text
    assert "@v5" not in workflow_text
    assert "@release/v1" not in workflow_text
