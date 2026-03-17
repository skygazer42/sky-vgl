from pathlib import Path
import re


FORBIDDEN_IMPORT_PATTERNS = (
    re.compile(r"^\s*from\s+vgl\.core(?:\.|\s+import)\b", re.MULTILINE),
    re.compile(r"^\s*import\s+vgl\.core(?:\.|\b)", re.MULTILINE),
    re.compile(r"^\s*from\s+vgl\.data(?:\.|\s+import)\b", re.MULTILINE),
    re.compile(r"^\s*import\s+vgl\.data(?:\.|\b)", re.MULTILINE),
    re.compile(r"^\s*from\s+vgl\.train(?:\.|\s+import)\b", re.MULTILINE),
    re.compile(r"^\s*import\s+vgl\.train(?:\.|\b)", re.MULTILINE),
)


def test_examples_and_integration_tests_prefer_domain_packages():
    project_root = Path(__file__).resolve().parents[1]
    files_to_check = [
        *sorted((project_root / "examples").rglob("*.py")),
        *sorted((project_root / "tests" / "integration").rglob("*.py")),
    ]

    offenders: list[str] = []
    for path in files_to_check:
        content = path.read_text()
        if any(pattern.search(content) for pattern in FORBIDDEN_IMPORT_PATTERNS):
            offenders.append(str(path.relative_to(project_root)))

    assert offenders == []
