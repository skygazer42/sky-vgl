#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import repo_script_imports

load_repo_module = repo_script_imports.load_repo_module


_CONTRACTS = None
_FORBIDDEN_IMPORT_PATTERNS: tuple[re.Pattern[str], ...] | None = None


def _bind_contracts():
    global _CONTRACTS
    global FORBIDDEN_PREFERRED_IMPORT_PREFIXES
    global PUBLIC_EXAMPLE_MODULES
    global public_surface_specs
    global _FORBIDDEN_IMPORT_PATTERNS
    global FORBIDDEN_IMPORT_PATTERNS
    if _CONTRACTS is None:
        _CONTRACTS = load_repo_module("scripts.contracts")
        FORBIDDEN_PREFERRED_IMPORT_PREFIXES = _CONTRACTS.FORBIDDEN_PREFERRED_IMPORT_PREFIXES
        PUBLIC_EXAMPLE_MODULES = _CONTRACTS.PUBLIC_EXAMPLE_MODULES
        public_surface_specs = _CONTRACTS.public_surface_specs
        _FORBIDDEN_IMPORT_PATTERNS = tuple(
            pattern
            for prefix in FORBIDDEN_PREFERRED_IMPORT_PREFIXES
            for pattern in (
                re.compile(rf"^\s*from\s+{re.escape(prefix)}(?:\.|\s+import)\b", re.MULTILINE),
                re.compile(rf"^\s*import\s+{re.escape(prefix)}(?:\.|\b)", re.MULTILINE),
            )
        )
        FORBIDDEN_IMPORT_PATTERNS = _FORBIDDEN_IMPORT_PATTERNS
    return _CONTRACTS


def __getattr__(name: str):
    if name in {
        "FORBIDDEN_PREFERRED_IMPORT_PREFIXES",
        "PUBLIC_EXAMPLE_MODULES",
        "public_surface_specs",
        "FORBIDDEN_IMPORT_PATTERNS",
    }:
        _bind_contracts()
        return globals()[name]
    raise AttributeError(name)


CheckFn = Callable[[], tuple[bool, str]]


@dataclass(frozen=True)
class ScanTask:
    id: str
    category: str
    description: str
    check: CheckFn


@dataclass(frozen=True)
class ListTaskSpec:
    id: str
    category: str
    description: str


@dataclass(frozen=True)
class StaticReexportSpec:
    category: str
    description: str
    relative_path: str
    symbol: str
    expected_source: str


@dataclass(frozen=True)
class ModuleSurface:
    imports: dict[str, str]
    exports: set[str]


class ScanContext:
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root.resolve()
        self._text_cache: dict[Path, str] = {}
        self._surface_cache: dict[Path, ModuleSurface] = {}

    def resolve(self, relative_path: str) -> Path:
        return self.repo_root / relative_path

    def read_text(self, relative_path: str) -> str:
        path = self.resolve(relative_path)
        cached = self._text_cache.get(path)
        if cached is None:
            cached = path.read_text(encoding="utf-8")
            self._text_cache[path] = cached
        return cached

    def module_surface(self, relative_path: str) -> ModuleSurface:
        path = self.resolve(relative_path)
        cached = self._surface_cache.get(path)
        if cached is not None:
            return cached

        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports: dict[str, str] = {}
        exports: set[str] = set()

        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    imports[local_name] = f"{node.module}.{alias.name}"
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        exports = _string_sequence(node.value)
                        break

        cached = ModuleSurface(imports=imports, exports=exports)
        self._surface_cache[path] = cached
        return cached

    def has_main_guard(self, relative_path: str) -> tuple[bool, str]:
        text = self.read_text(relative_path)
        has_guard = 'if __name__ == "__main__":' in text or "if __name__ == '__main__':" in text
        return has_guard, relative_path

    def scan_tree_for_forbidden_imports(self, relative_dir: str) -> tuple[bool, str]:
        _bind_contracts()
        root = self.resolve(relative_dir)
        offenders: list[str] = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            if any(pattern.search(text) for pattern in FORBIDDEN_IMPORT_PATTERNS):
                offenders.append(str(path.relative_to(self.repo_root)))
        if offenders:
            return False, ", ".join(offenders[:5])
        return True, f"{relative_dir} has no vgl.core/vgl.data/vgl.train imports"


def _string_sequence(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set()
    values: set[str] = set()
    for item in node.elts:
        if isinstance(item, ast.Constant) and isinstance(item.value, str):
            values.add(item.value)
    return values


def _named_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value
            continue
        if isinstance(node, ast.AnnAssign) and node.value is not None and isinstance(node.target, ast.Name):
            assignments[node.target.id] = node.value
    return assignments


def _ordered_string_sequence(
    node: ast.AST,
    assignments: dict[str, ast.AST] | None = None,
    seen: set[str] | None = None,
) -> tuple[str, ...]:
    if seen is None:
        seen = set()
    if isinstance(node, ast.Name):
        if assignments is None or node.id not in assignments:
            raise RuntimeError(f"expected string sequence binding for {node.id!r} in scripts/contracts.py")
        if node.id in seen:
            raise RuntimeError(f"scripts/contracts.py contains a recursive string sequence reference: {node.id}")
        return _ordered_string_sequence(assignments[node.id], assignments, seen | {node.id})
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _ordered_string_sequence(node.left, assignments, seen) + _ordered_string_sequence(
            node.right,
            assignments,
            seen,
        )
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise RuntimeError("expected a tuple, list, or named string sequence in scripts/contracts.py")
    values: list[str] = []
    for item in node.elts:
        values.append(_string_literal(item, assignments, seen))
    return tuple(values)


def _static_reexport_specs(node: ast.AST) -> tuple[StaticReexportSpec, ...]:
    if not isinstance(node, ast.Tuple):
        raise RuntimeError("expected a tuple of ReexportSpec entries in scripts/contracts.py")

    specs: list[StaticReexportSpec] = []
    for item in node.elts:
        if not isinstance(item, ast.Call) or not isinstance(item.func, ast.Name) or item.func.id != "ReexportSpec":
            raise RuntimeError("expected ReexportSpec(...) entries in scripts/contracts.py")
        if item.keywords:
            raise RuntimeError("expected positional ReexportSpec arguments in scripts/contracts.py")
        values = _ordered_string_sequence(ast.Tuple(elts=item.args))
        if len(values) != 5:
            raise RuntimeError("expected five string arguments for ReexportSpec(...)")
        specs.append(StaticReexportSpec(*values))
    return tuple(specs)


def _public_surface_spec_group_names(tree: ast.Module) -> tuple[str, ...]:
    assignments = _named_assignments(tree)
    return_value: ast.AST | None = None
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "public_surface_specs":
            continue
        for statement in node.body:
            if isinstance(statement, ast.Return):
                return_value = statement.value
                break
        break
    if return_value is None:
        raise RuntimeError("scripts/contracts.py missing public_surface_specs() return")
    return _return_name_chain(return_value, assignments, set())


def _return_name_chain(
    node: ast.AST | None,
    assignments: dict[str, ast.AST],
    seen: set[str],
) -> tuple[str, ...]:
    if node is None:
        raise RuntimeError("public_surface_specs() must return named reexport groups")
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _return_name_chain(node.left, assignments, seen) + _return_name_chain(node.right, assignments, seen)
    if isinstance(node, ast.Name):
        if node.id in seen:
            raise RuntimeError(f"public_surface_specs() contains a recursive group reference: {node.id}")
        assigned = assignments.get(node.id)
        if assigned is None or _is_reexport_group_node(assigned):
            return (node.id,)
        return _return_name_chain(assigned, assignments, seen | {node.id})
    raise RuntimeError("public_surface_specs() must concatenate named reexport groups")


def _string_literal(
    node: ast.AST,
    assignments: dict[str, ast.AST] | None = None,
    seen: set[str] | None = None,
) -> str:
    if seen is None:
        seen = set()
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        if assignments is None or node.id not in assignments:
            raise RuntimeError(f"expected string binding for {node.id!r} in scripts/contracts.py")
        if node.id in seen:
            raise RuntimeError(f"scripts/contracts.py contains a recursive string reference: {node.id}")
        return _string_literal(assignments[node.id], assignments, seen | {node.id})
    raise RuntimeError("expected only string literals or named string bindings in scripts/contracts.py")


def _is_reexport_group_node(node: ast.AST) -> bool:
    if not isinstance(node, ast.Tuple):
        return False
    return all(
        isinstance(item, ast.Call) and isinstance(item.func, ast.Name) and item.func.id == "ReexportSpec"
        for item in node.elts
    )


def _list_catalog(repo_root: Path) -> tuple[tuple[StaticReexportSpec, ...], tuple[str, ...]]:
    contracts_path = repo_root / "scripts" / "contracts.py"
    tree = ast.parse(contracts_path.read_text(encoding="utf-8"), filename=str(contracts_path))
    spec_group_names = _public_surface_spec_group_names(tree)
    assignments = _named_assignments(tree)

    spec_groups: dict[str, tuple[StaticReexportSpec, ...]] = {}
    public_example_modules: tuple[str, ...] | None = None
    for name, value in assignments.items():
        if name == "PUBLIC_EXAMPLE_MODULES":
            public_example_modules = _ordered_string_sequence(value, assignments)
        elif name in spec_group_names:
            spec_groups[name] = _static_reexport_specs(value)

    missing_groups = [name for name in spec_group_names if name not in spec_groups]
    if missing_groups:
        raise RuntimeError(f"scripts/contracts.py missing public surface groups: {', '.join(missing_groups)}")
    if public_example_modules is None:
        raise RuntimeError("scripts/contracts.py missing PUBLIC_EXAMPLE_MODULES")

    reexport_specs = tuple(spec for group in spec_group_names for spec in spec_groups[group])
    return reexport_specs, public_example_modules


def list_task_specs(repo_root: Path) -> list[ListTaskSpec]:
    reexport_specs, public_example_modules = _list_catalog(repo_root.resolve())

    tasks = [
        ListTaskSpec(f"{index:03d}", spec.category, spec.description)
        for index, spec in enumerate(reexport_specs, start=1)
    ]
    next_id = len(tasks) + 1
    tasks.extend(
        ListTaskSpec(f"{index:03d}", "example", f"{relative_path} has __main__ guard")
        for index, relative_path in enumerate(public_example_modules, start=next_id)
    )
    next_id = len(tasks) + 1
    tasks.append(ListTaskSpec(f"{next_id:03d}", "imports", "examples avoids legacy import paths"))
    tasks.append(ListTaskSpec(f"{next_id + 1:03d}", "imports", "tests/integration avoids legacy import paths"))
    return tasks


def _reexport_task(
    ctx: ScanContext,
    task_id: str,
    category: str,
    description: str,
    relative_path: str,
    symbol: str,
    expected_source: str,
) -> ScanTask:
    def check() -> tuple[bool, str]:
        surface = ctx.module_surface(relative_path)
        imported_source = surface.imports.get(symbol)
        if imported_source != expected_source:
            return False, f"{relative_path} imports {symbol!r} from {imported_source!r}"
        if symbol not in surface.exports:
            return False, f"{relative_path} __all__ missing {symbol!r}"
        return True, f"{relative_path} exports {symbol} from {expected_source}"

    return ScanTask(task_id, category, description, check)


def _main_guard_task(ctx: ScanContext, task_id: str, relative_path: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        path = ctx.resolve(relative_path)
        if not path.exists():
            return False, f"{relative_path} missing"
        return ctx.has_main_guard(relative_path)

    return ScanTask(task_id, "example", f"{relative_path} has __main__ guard", check)


def _forbidden_import_task(ctx: ScanContext, task_id: str, relative_dir: str) -> ScanTask:
    def check() -> tuple[bool, str]:
        return ctx.scan_tree_for_forbidden_imports(relative_dir)

    return ScanTask(task_id, "imports", f"{relative_dir} avoids legacy import paths", check)


def build_tasks(repo_root: Path) -> list[ScanTask]:
    _bind_contracts()
    ctx = ScanContext(repo_root)
    tasks = [
        _reexport_task(
            ctx,
            f"{index:03d}",
            spec.category,
            spec.description,
            spec.relative_path,
            spec.symbol,
            spec.expected_source,
        )
        for index, spec in enumerate(public_surface_specs(), start=1)
    ]
    next_id = len(tasks) + 1
    tasks.extend(
        _main_guard_task(ctx, f"{index:03d}", relative_path)
        for index, relative_path in enumerate(PUBLIC_EXAMPLE_MODULES, start=next_id)
    )
    next_id = len(tasks) + 1
    tasks.append(_forbidden_import_task(ctx, f"{next_id:03d}", "examples"))
    tasks.append(_forbidden_import_task(ctx, f"{next_id + 1:03d}", "tests/integration"))

    expected_task_count = len(public_surface_specs()) + len(PUBLIC_EXAMPLE_MODULES) + 2
    if len(tasks) != expected_task_count:
        raise RuntimeError(f"expected {expected_task_count} scan tasks, found {len(tasks)}")
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
    parser = argparse.ArgumentParser(description="Scan public import/export and example script contracts.")
    parser.add_argument("--list", action="store_true", help="List scan tasks without executing them.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to scan. Defaults to this checkout root.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    if args.list:
        for task in list_task_specs(repo_root):
            print(f"SCAN {task.id} [{task.category}] {task.description}")
        return 0
    tasks = build_tasks(repo_root)
    return run_tasks(tasks)


if __name__ == "__main__":
    raise SystemExit(main())
