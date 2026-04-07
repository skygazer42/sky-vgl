from __future__ import annotations

from pathlib import Path


def workflow_job_text_from_text(text: str, job_name: str, *, source: str = "workflow") -> str:
    lines = text.splitlines()
    jobs_start = None
    jobs_end = len(lines)

    for index, line in enumerate(lines):
        if line == "jobs:":
            jobs_start = index + 1
            break

    if jobs_start is None:
        raise KeyError(f"{source} missing 'jobs:' section")

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
        raise KeyError(f"{source} missing workflow job {job_name!r}")

    end = jobs_end
    for index in range(start + 1, jobs_end):
        line = lines[index]
        if line.startswith("  ") and not line.startswith("    ") and line.rstrip().endswith(":"):
            end = index
            break

    return "\n".join(lines[start:end]) + "\n"


def workflow_job_text(path: Path, job_name: str) -> str:
    return workflow_job_text_from_text(path.read_text(encoding="utf-8"), job_name, source=str(path))


def workflow_job_contains_text(
    text: str,
    job_name: str,
    snippet: str,
    *,
    source: str = "workflow",
) -> tuple[bool, str]:
    try:
        job_text = workflow_job_text_from_text(text, job_name, source=source)
    except KeyError as exc:
        return False, str(exc)
    return snippet in job_text, f"{source} job {job_name!r} contains {snippet!r}"
