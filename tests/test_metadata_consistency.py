import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "metadata_consistency.py"


def test_metadata_consistency_passes_on_repository():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY 6/6 passed" in completed.stdout


def test_metadata_consistency_validates_tag_refs():
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--git-ref", "refs/tags/v999.0.0"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "tag version" in completed.stdout
