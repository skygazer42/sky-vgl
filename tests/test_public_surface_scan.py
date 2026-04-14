import importlib.util
import subprocess
import sys
import textwrap
from pathlib import Path

import scripts.contracts as contracts
from scripts.contracts import PUBLIC_EXAMPLE_MODULES, ROOT_EXPORT_POLICY_DESCRIPTION, public_surface_specs


REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_SCRIPT = REPO_ROOT / "scripts" / "public_surface_scan.py"


def _load_public_surface_scan(module_name: str = "public_surface_scan"):
    spec = importlib.util.spec_from_file_location(module_name, SCAN_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_public_surface_scan_prefers_repo_contracts_module(monkeypatch, tmp_path):
    shadow_module = tmp_path / "contracts.py"
    shadow_module.write_text(
        textwrap.dedent(
            """
            FORBIDDEN_PREFERRED_IMPORT_PREFIXES = ()
            PUBLIC_EXAMPLE_MODULES = ()

            def public_surface_specs():
                return ()
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    shadowed = sys.modules.pop("contracts", None)
    try:
        public_surface_scan = _load_public_surface_scan("public_surface_scan_shadowed")
    finally:
        sys.modules.pop("public_surface_scan_shadowed", None)
        if shadowed is not None:
            sys.modules["contracts"] = shadowed

    assert public_surface_scan.PUBLIC_EXAMPLE_MODULES == contracts.PUBLIC_EXAMPLE_MODULES
    assert public_surface_scan.public_surface_specs() == contracts.public_surface_specs()


def test_public_surface_scan_lists_stable_catalog():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT), "--list"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    listed = [line for line in completed.stdout.splitlines() if line.startswith("SCAN ")]
    assert len(listed) == len(public_surface_specs()) + len(PUBLIC_EXAMPLE_MODULES) + 3
    assert any("vgl.data reexports PlanStage from vgl.dataloading" in line for line in listed)
    assert any("vgl.data.plan reexports PlanStage from vgl.dataloading.plan" in line for line in listed)
    assert any("vgl.data.materialize reexports materialize_batch from vgl.dataloading.materialize" in line for line in listed)
    assert any("vgl.data.sampler reexports RandomWalkSampler from vgl.dataloading.advanced" in line for line in listed)
    assert any("vgl.data.sampler reexports GraphSAINTEdgeSampler from vgl.dataloading.advanced" in line for line in listed)
    assert any("vgl.data.transform reexports BaseTransform from vgl.transforms" in line for line in listed)
    assert any("vgl.data.transform reexports NormalizeFeatures from vgl.transforms" in line for line in listed)
    assert any("vgl.data reexports ClusterData from vgl.dataloading" in line for line in listed)
    assert any("vgl.data reexports RandomWalkSampler from vgl.dataloading" in line for line in listed)
    assert any("vgl.data reexports GraphSAINTEdgeSampler from vgl.dataloading" in line for line in listed)
    assert any("vgl.data reexports GraphSAINTRandomWalkSampler from vgl.dataloading" in line for line in listed)
    assert any("vgl.train reexports ModelCheckpoint from vgl.engine" in line for line in listed)
    assert any("vgl.train reexports HitsAtK from vgl.metrics" in line for line in listed)
    assert any("vgl.train reexports FloodingTask from vgl.tasks" in line for line in listed)
    assert any("vgl.core reexports EdgeStore from vgl.graph" in line for line in listed)
    assert any("vgl.core reexports SchemaError from vgl.graph" in line for line in listed)
    assert any(ROOT_EXPORT_POLICY_DESCRIPTION in line for line in listed)


def test_public_surface_scan_list_catalog_follows_public_surface_specs_function(tmp_path: Path):
    repo_root = tmp_path / "repo"
    contracts_path = repo_root / "scripts" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            PUBLIC_EXAMPLE_MODULES = ("examples/demo.py",)


            @dataclass(frozen=True)
            class ReexportSpec:
                category: str
                description: str
                relative_path: str
                symbol: str
                expected_source: str


            CUSTOM_EXPORTS = (
                ReexportSpec(
                    "root",
                    "demo exports Widget from demo.widget",
                    "demo/__init__.py",
                    "Widget",
                    "demo.widget.Widget",
                ),
            )


            def public_surface_specs():
                return CUSTOM_EXPORTS
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    scan = _load_public_surface_scan("public_surface_scan_catalog_probe")
    try:
        listed_tasks = scan.list_task_specs(repo_root)
    finally:
        sys.modules.pop("public_surface_scan_catalog_probe", None)

    assert [task.description for task in listed_tasks] == [
        "demo exports Widget from demo.widget",
        "examples/demo.py has __main__ guard",
        "examples avoids legacy import paths",
        "tests/integration avoids legacy import paths",
    ]


def test_public_surface_scan_list_catalog_follows_named_group_composition(tmp_path: Path):
    repo_root = tmp_path / "repo"
    contracts_path = repo_root / "scripts" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            PUBLIC_EXAMPLE_MODULES = ()


            @dataclass(frozen=True)
            class ReexportSpec:
                category: str
                description: str
                relative_path: str
                symbol: str
                expected_source: str


            ROOT_EXPORTS = (
                ReexportSpec(
                    "root",
                    "demo exports Widget from demo.widget",
                    "demo/__init__.py",
                    "Widget",
                    "demo.widget.Widget",
                ),
            )
            LEGACY_EXPORTS = (
                ReexportSpec(
                    "legacy",
                    "legacy demo exports LegacyWidget from demo.legacy",
                    "demo/legacy.py",
                    "LegacyWidget",
                    "demo.legacy.LegacyWidget",
                ),
            )
            ALL_EXPORTS = ROOT_EXPORTS + LEGACY_EXPORTS


            def public_surface_specs():
                return ALL_EXPORTS
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    scan = _load_public_surface_scan("public_surface_scan_group_chain_probe")
    try:
        listed_tasks = scan.list_task_specs(repo_root)
    finally:
        sys.modules.pop("public_surface_scan_group_chain_probe", None)

    assert [task.description for task in listed_tasks] == [
        "demo exports Widget from demo.widget",
        "legacy demo exports LegacyWidget from demo.legacy",
        "examples avoids legacy import paths",
        "tests/integration avoids legacy import paths",
    ]


def test_public_surface_scan_list_catalog_follows_named_example_module_composition(tmp_path: Path):
    repo_root = tmp_path / "repo"
    contracts_path = repo_root / "scripts" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            CORE_EXAMPLES = ("examples/core_demo.py",)
            EXTRA_EXAMPLES = ("examples/extra_demo.py",)
            PUBLIC_EXAMPLE_MODULES = CORE_EXAMPLES + EXTRA_EXAMPLES


            @dataclass(frozen=True)
            class ReexportSpec:
                category: str
                description: str
                relative_path: str
                symbol: str
                expected_source: str


            ROOT_EXPORTS = (
                ReexportSpec(
                    "root",
                    "demo exports Widget from demo.widget",
                    "demo/__init__.py",
                    "Widget",
                    "demo.widget.Widget",
                ),
            )


            def public_surface_specs():
                return ROOT_EXPORTS
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    scan = _load_public_surface_scan("public_surface_scan_examples_probe")
    try:
        listed_tasks = scan.list_task_specs(repo_root)
    finally:
        sys.modules.pop("public_surface_scan_examples_probe", None)

    assert [task.description for task in listed_tasks] == [
        "demo exports Widget from demo.widget",
        "examples/core_demo.py has __main__ guard",
        "examples/extra_demo.py has __main__ guard",
        "examples avoids legacy import paths",
        "tests/integration avoids legacy import paths",
    ]


def test_public_surface_scan_list_catalog_supports_annotated_assignments(tmp_path: Path):
    repo_root = tmp_path / "repo"
    contracts_path = repo_root / "scripts" / "contracts.py"
    contracts_path.parent.mkdir(parents=True)
    contracts_path.write_text(
        textwrap.dedent(
            """
            from dataclasses import dataclass

            EXAMPLES: tuple[str, ...] = ("examples/demo.py",)
            PUBLIC_EXAMPLE_MODULES: tuple[str, ...] = EXAMPLES


            @dataclass(frozen=True)
            class ReexportSpec:
                category: str
                description: str
                relative_path: str
                symbol: str
                expected_source: str


            ROOT_EXPORTS: tuple[ReexportSpec, ...] = (
                ReexportSpec(
                    "root",
                    "demo exports Widget from demo.widget",
                    "demo/__init__.py",
                    "Widget",
                    "demo.widget.Widget",
                ),
            )


            def public_surface_specs() -> tuple[ReexportSpec, ...]:
                return ROOT_EXPORTS
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    scan = _load_public_surface_scan("public_surface_scan_annotated_probe")
    try:
        listed_tasks = scan.list_task_specs(repo_root)
    finally:
        sys.modules.pop("public_surface_scan_annotated_probe", None)

    assert [task.description for task in listed_tasks] == [
        "demo exports Widget from demo.widget",
        "examples/demo.py has __main__ guard",
        "examples avoids legacy import paths",
        "tests/integration avoids legacy import paths",
    ]


def test_public_surface_scan_passes_on_repository():
    completed = subprocess.run(
        [sys.executable, str(SCAN_SCRIPT)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    expected = len(public_surface_specs()) + len(PUBLIC_EXAMPLE_MODULES) + 3
    assert f"SUMMARY {expected}/{expected} passed" in completed.stdout
    assert ROOT_EXPORT_POLICY_DESCRIPTION in completed.stdout
