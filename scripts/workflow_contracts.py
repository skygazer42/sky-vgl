from __future__ import annotations

import re
from pathlib import Path


_JOBS_SECTION_HEADER_RE = re.compile(r"^jobs:\s*(?:#.*)?$")
_MAPPING_HEADER_RE = re.compile(r"^  (?! )(?:\"[^\"]+\"|'[^']+'|[^:#\s][^:]*):\s*(?:#.*)?$")


def _is_jobs_section_header(line: str) -> bool:
    return _JOBS_SECTION_HEADER_RE.match(line) is not None


def _job_header_pattern(job_name: str) -> re.Pattern[str]:
    escaped = re.escape(job_name)
    return re.compile(rf"^  (?! )(?:\"{escaped}\"|'{escaped}'|{escaped}):\s*(?:#.*)?$")


def workflow_job_text_from_text(text: str, job_name: str, *, source: str = "workflow") -> str:
    lines = text.splitlines()
    jobs_start = None
    jobs_end = len(lines)

    for index, line in enumerate(lines):
        if _is_jobs_section_header(line):
            jobs_start = index + 1
            break

    if jobs_start is None:
        raise KeyError(f"{source} missing 'jobs:' section")

    for index in range(jobs_start, len(lines)):
        line = lines[index]
        if line and not line.startswith((" ", "#")):
            jobs_end = index
            break

    header = _job_header_pattern(job_name)

    start = None
    for index in range(jobs_start, jobs_end):
        line = lines[index]
        if header.match(line):
            start = index
            break

    if start is None:
        raise KeyError(f"{source} missing workflow job {job_name!r}")

    end = jobs_end
    for index in range(start + 1, jobs_end):
        line = lines[index]
        if _MAPPING_HEADER_RE.match(line):
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


def workflow_step_text_from_text(
    text: str,
    job_name: str,
    step_name: str,
    *,
    source: str = "workflow",
) -> str:
    lines = workflow_job_text_from_text(text, job_name, source=source).splitlines()
    header = f"      - name: {step_name}"

    start = None
    for index, line in enumerate(lines):
        if line == header:
            start = index
            break

    if start is None:
        raise KeyError(f"{source} job {job_name!r} missing workflow step {step_name!r}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if line.startswith("      - "):
            end = index
            break

    return "\n".join(lines[start:end]) + "\n"


def workflow_step_text(path: Path, job_name: str, step_name: str) -> str:
    return workflow_step_text_from_text(
        path.read_text(encoding="utf-8"),
        job_name,
        step_name,
        source=str(path),
    )


def workflow_step_contains_text(
    text: str,
    job_name: str,
    step_name: str,
    snippet: str,
    *,
    source: str = "workflow",
) -> tuple[bool, str]:
    try:
        step_text = workflow_step_text_from_text(text, job_name, step_name, source=source)
    except KeyError as exc:
        return False, str(exc)
    return snippet in step_text, f"{source} job {job_name!r} step {step_name!r} contains {snippet!r}"
