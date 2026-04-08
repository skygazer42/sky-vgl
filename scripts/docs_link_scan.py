#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fnmatch
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse


CheckFn = Callable[[], tuple[bool, str]]

LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
IMG_SRC_PATTERN = re.compile(r'src="([^"]+)"')
RAW_REPO_PREFIX = ("raw.githubusercontent.com", "skygazer42", "sky-vgl", "main")


@dataclass(frozen=True)
class ScanTask:
    id: str
    category: str
    description: str
    check: CheckFn


class DocsContext:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self._anchor_cache: dict[Path, set[str]] = {}
        self._public_docs_cache: list[Path] | None = None

    def public_docs(self) -> list[Path]:
        cached = self._public_docs_cache
        if cached is None:
            docs_root = self.repo_root / "docs"
            excluded_patterns = _mkdocs_not_in_nav_patterns(self.repo_root)
            cached = [self.repo_root / "README.md"]
            for path in sorted(docs_root.rglob("*.md")):
                relative_doc = path.relative_to(docs_root).as_posix()
                if any(fnmatch.fnmatch(relative_doc, pattern) for pattern in excluded_patterns):
                    continue
                cached.append(path)
            self._public_docs_cache = cached
        return cached

    def anchors_for(self, path: Path) -> set[str]:
        path = path.resolve()
        cached = self._anchor_cache.get(path)
        if cached is None:
            text = path.read_text(encoding="utf-8")
            anchors = set()
            for line in text.splitlines():
                if not line.startswith("#"):
                    continue
                heading = line.lstrip("#").strip()
                if not heading:
                    continue
                slug = re.sub(r"[^\w\s-]", "", heading.lower())
                slug = re.sub(r"\s+", "-", slug).strip("-")
                if slug:
                    anchors.add(slug)
            cached = anchors
            self._anchor_cache[path] = cached
        return cached


def _mkdocs_not_in_nav_patterns(repo_root: Path) -> tuple[str, ...]:
    mkdocs_path = repo_root / "mkdocs.yml"
    if not mkdocs_path.is_file():
        return ()

    patterns: list[str] = []
    in_block = False
    for line in mkdocs_path.read_text(encoding="utf-8").splitlines():
        if not in_block:
            if line.startswith("not_in_nav:"):
                payload = line.partition(":")[2].strip()
                if payload:
                    if payload in {"|", ">"}:
                        in_block = True
                        continue
                    if payload.startswith("[") and payload.endswith("]"):
                        patterns.extend(_flow_sequence_patterns(payload))
                    else:
                        patterns.append(_strip_optional_quotes(payload))
                    break
                in_block = True
            continue
        if line and not line.startswith(" "):
            break
        pattern = line.strip()
        if pattern.startswith("- "):
            pattern = pattern[2:].strip()
        pattern = _strip_optional_quotes(pattern)
        if pattern:
            patterns.append(pattern)
    return tuple(patterns)


def _flow_sequence_patterns(payload: str) -> tuple[str, ...]:
    items: list[str] = []
    current: list[str] = []
    quote: str | None = None
    for char in payload[1:-1]:
        if quote is not None:
            if char == quote:
                quote = None
            current.append(char)
            continue
        if char in {'"', "'"}:
            quote = char
            current.append(char)
            continue
        if char == ",":
            item = _strip_optional_quotes("".join(current).strip())
            if item:
                items.append(item)
            current = []
            continue
        current.append(char)
    tail = _strip_optional_quotes("".join(current).strip())
    if tail:
        items.append(tail)
    return tuple(items)


def _strip_optional_quotes(pattern: str) -> str:
    if len(pattern) >= 2 and pattern[0] == pattern[-1] and pattern[0] in {'"', "'"}:
        return pattern[1:-1]
    return pattern


def _same_doc_anchor_task(ctx: DocsContext, task_id: str, source: Path, fragment: str) -> ScanTask:
    source_rel = source.relative_to(ctx.repo_root)

    def check() -> tuple[bool, str]:
        anchors = ctx.anchors_for(source)
        return fragment in anchors, f"#{fragment} in {source_rel}"

    return ScanTask(task_id, "anchor", f"{source_rel} anchor #{fragment} resolves", check)


def _local_link_task(ctx: DocsContext, task_id: str, source: Path, target: str) -> ScanTask:
    source_rel = source.relative_to(ctx.repo_root)

    def check() -> tuple[bool, str]:
        parsed = urlparse(target)
        candidate = (source.parent / unquote(parsed.path)).resolve()
        try:
            candidate_rel = candidate.relative_to(ctx.repo_root)
        except ValueError:
            return False, f"{target!r} escapes repository root"
        if not candidate.exists():
            return False, f"{candidate_rel} missing"
        if parsed.fragment:
            if candidate.suffix.lower() != ".md":
                return False, f"{target!r} uses an anchor on a non-markdown target"
            anchors = ctx.anchors_for(candidate)
            if parsed.fragment not in anchors:
                return False, f"#{parsed.fragment} missing in {candidate_rel}"
        return True, f"{source_rel} -> {candidate_rel}"

    return ScanTask(task_id, "link", f"{source_rel} link {target} resolves", check)


def _raw_asset_task(ctx: DocsContext, task_id: str, source: Path, target: str) -> ScanTask:
    source_rel = source.relative_to(ctx.repo_root)

    def check() -> tuple[bool, str]:
        parsed = urlparse(target)
        if parsed.netloc != RAW_REPO_PREFIX[0]:
            return False, f"{target} is not a raw.githubusercontent.com URL"
        parts = [part for part in parsed.path.split("/") if part]
        if len(parts) < 4:
            return False, f"{target} has incomplete raw asset path"
        owner, repo, branch = parts[:3]
        if (owner, repo, branch) != RAW_REPO_PREFIX[1:]:
            return False, f"{target} does not point at skygazer42/sky-vgl main"
        asset_rel = Path(*parts[3:])
        asset_path = (ctx.repo_root / asset_rel).resolve()
        try:
            asset_path.relative_to(ctx.repo_root)
        except ValueError:
            return False, f"{target} escapes repository root"
        return asset_path.is_file(), f"{source_rel} -> {asset_rel}"

    return ScanTask(task_id, "asset", f"{source_rel} raw asset {target} resolves", check)


def build_tasks(repo_root: Path) -> list[ScanTask]:
    ctx = DocsContext(repo_root)
    tasks: list[ScanTask] = []
    next_id = 1

    for source in ctx.public_docs():
        text = source.read_text(encoding="utf-8")
        for target in LINK_PATTERN.findall(text):
            parsed = urlparse(target)
            if target.startswith("#"):
                tasks.append(_same_doc_anchor_task(ctx, f"{next_id:03d}", source, target[1:]))
                next_id += 1
                continue
            if parsed.scheme in {"http", "https", "mailto"}:
                continue
            tasks.append(_local_link_task(ctx, f"{next_id:03d}", source, target))
            next_id += 1

        for target in IMG_SRC_PATTERN.findall(text):
            parsed = urlparse(target)
            if parsed.scheme not in {"http", "https"}:
                continue
            if parsed.netloc != RAW_REPO_PREFIX[0]:
                continue
            tasks.append(_raw_asset_task(ctx, f"{next_id:03d}", source, target))
            next_id += 1

    if not tasks:
        raise RuntimeError("expected at least one docs link scan task")
    return tasks


def run_tasks(tasks: list[ScanTask]) -> int:
    passed = 0
    for task in tasks:
        ok, detail = task.check()
        status = "PASS" if ok else "FAIL"
        print(f"{status} {task.id} [{task.category}] {task.description} :: {detail}")
        if ok:
            passed += 1
    print(f"SUMMARY {passed}/{len(tasks)} passed")
    return 0 if passed == len(tasks) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan README and public docs for broken local links.")
    parser.add_argument("--list", action="store_true", help="List scan tasks without executing them.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan. Defaults to this checkout root.",
    )
    args = parser.parse_args()

    tasks = build_tasks(args.repo_root.resolve())
    if args.list:
        for task in tasks:
            print(f"SCAN {task.id} [{task.category}] {task.description}")
        return 0
    return run_tasks(tasks)


if __name__ == "__main__":
    raise SystemExit(main())
