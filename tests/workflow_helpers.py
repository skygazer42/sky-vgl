from __future__ import annotations

from pathlib import Path


def workflow_job_text(path: Path, job_name: str) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    jobs_start = None
    jobs_end = len(lines)

    for index, line in enumerate(lines):
        if line == "jobs:":
            jobs_start = index + 1
            break

    if jobs_start is None:
        raise AssertionError(f"'jobs:' section not found in {path}")

    for index in range(jobs_start, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "#")):
            jobs_end = index
            break

    header = f"  {job_name}:"

    start = None
    for index in range(jobs_start, jobs_end):
        line = lines[index]
        if line == header:
            start = index
            break

    if start is None:
        raise AssertionError(f"workflow job {job_name!r} not found in {path}")

    end = jobs_end
    for index in range(start + 1, jobs_end):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":"):
            end = index
            break

    return "\n".join(lines[start:end]) + "\n"
