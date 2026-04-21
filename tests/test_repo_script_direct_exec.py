import json
import subprocess
import sys
import textwrap
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script_outside_repo(tmp_path: Path, relative_script: str, *args: str) -> subprocess.CompletedProcess[str]:
    script = REPO_ROOT / relative_script
    return subprocess.run(
        [sys.executable, str(script), *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )


def test_full_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/full_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [repo] README.md exists" in completed.stdout
    assert "SUMMARY listed " in completed.stdout


def test_metadata_consistency_runs_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/metadata_consistency.py")

    assert completed.returncode == 0, completed.stderr
    assert "SUMMARY 6/6 passed" in completed.stdout


def test_public_surface_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/public_surface_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001" in completed.stdout
    assert "tests/integration avoids legacy import paths" in completed.stdout


def test_interop_smoke_lists_backends_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/interop_smoke.py", "--list-backends")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["pyg", "dgl"]


def test_release_contract_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/release_contract_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [artifact] built wheel exists" in completed.stdout
    assert "SCAN 036 [sdist] sdist excludes __pycache__" in completed.stdout


def test_install_release_extras_help_renders_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/install_release_extras.py", "--help")

    assert completed.returncode == 0
    assert "usage: install_release_extras.py" in completed.stdout
    assert "--artifact-dir" in completed.stdout


def test_install_release_extras_help_does_not_require_packaging(tmp_path):
    script = REPO_ROOT / "scripts" / "install_release_extras.py"
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import runpy
        import sys

        class _PackagingBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "packaging" or fullname.startswith("packaging."):
                    raise ModuleNotFoundError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, _PackagingBlocker())
        sys.path.insert(0, {str(REPO_ROOT)!r})
        sys.argv = [sys.argv[1], "--help"]
        runpy.run_path(sys.argv[0], run_name="__main__")
        """
    ).strip()
    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage: install_release_extras.py" in completed.stdout
    assert "--artifact-dir" in completed.stdout


def test_release_smoke_help_renders_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/release_smoke.py", "--help")

    assert completed.returncode == 0
    assert "usage: release_smoke.py" in completed.stdout
    assert "--interop-backend" in completed.stdout


def test_docs_link_scan_lists_tasks_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/docs_link_scan.py", "--list")

    assert completed.returncode == 0, completed.stderr
    assert "README.md link LICENSE resolves" in completed.stdout
    assert "docs/getting-started/installation.md link ../support-matrix.md resolves" in completed.stdout


def test_dependency_audit_prints_requirements_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(
        tmp_path,
        "scripts/dependency_audit.py",
        "--print-requirements",
        "--groups",
        "runtime",
        "full",
    )

    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.splitlines()
    assert "torch>=2.4" in lines
    assert "typing_extensions>=4.12" in lines
    assert "numpy>=1.26" in lines
    assert "scipy>=1.11" in lines
    assert "networkx>=3.2" in lines
    assert "tensorboard>=2.14" in lines
    assert "dgl>=1.1.3,<2" in lines
    assert "torch-geometric>=2.5" in lines


def test_dependency_audit_can_parse_pyproject_without_tomli_modules(tmp_path):
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import runpy
        import sys

        class _TomlBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {{"tomli", "tomllib"}}:
                    raise ModuleNotFoundError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, _TomlBlocker())
        sys.path.insert(0, {str(REPO_ROOT)!r})
        script = sys.argv[1]
        sys.argv = [script, "--print-requirements", "--groups", "runtime"]
        runpy.run_path(script, run_name="__main__")
        """
    ).strip()

    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, str(REPO_ROOT / "scripts" / "dependency_audit.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "torch>=2.4" in completed.stdout.splitlines()


def test_extras_smoke_lists_defaults_outside_repo_root(tmp_path):
    completed = _run_script_outside_repo(tmp_path, "scripts/extras_smoke.py", "--list-defaults")

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["networkx", "scipy", "tensorboard"]


def test_non_parsing_script_paths_do_not_require_toml_parser(tmp_path):
    scripts_and_args = [
        ("scripts/dependency_audit.py", "--help"),
        ("scripts/full_scan.py", "--help"),
        ("scripts/metadata_consistency.py", "--help"),
        ("scripts/release_contract_scan.py", "--help"),
    ]
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import runpy
        import sys

        class _TomlBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {{"tomli", "tomllib"}}:
                    raise ModuleNotFoundError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, _TomlBlocker())
        sys.path.insert(0, {str(REPO_ROOT)!r})
        script, *args = sys.argv[1:]
        sys.argv = [script, *args]
        runpy.run_path(script, run_name="__main__")
        """
    ).strip()

    for relative_script, arg in scripts_and_args:
        completed = subprocess.run(
            [sys.executable, "-c", bootstrap, str(REPO_ROOT / relative_script), arg],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr


def test_release_contract_scan_list_does_not_require_toml_parser(tmp_path):
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import runpy
        import sys

        class _TomlBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname in {{"tomli", "tomllib"}}:
                    raise ModuleNotFoundError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, _TomlBlocker())
        sys.path.insert(0, {str(REPO_ROOT)!r})
        script = sys.argv[1]
        sys.argv = [script, "--list"]
        runpy.run_path(script, run_name="__main__")
        """
    ).strip()

    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, str(REPO_ROOT / "scripts" / "release_contract_scan.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [artifact] built wheel exists" in completed.stdout


def test_public_surface_scan_list_does_not_require_repo_contracts(tmp_path):
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import runpy
        import sys

        class _ContractsBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "scripts.contracts" or fullname.startswith("scripts.contracts."):
                    raise ModuleNotFoundError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, _ContractsBlocker())
        sys.path.insert(0, {str(REPO_ROOT)!r})
        script = sys.argv[1]
        sys.argv = [script, "--list"]
        runpy.run_path(script, run_name="__main__")
        """
    ).strip()

    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, str(REPO_ROOT / "scripts" / "public_surface_scan.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [root] vgl exports Graph from vgl.graph" in completed.stdout


def test_release_contract_scan_list_does_not_require_repo_contracts(tmp_path):
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import runpy
        import sys

        class _ContractsBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "scripts.contracts" or fullname.startswith("scripts.contracts."):
                    raise ModuleNotFoundError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, _ContractsBlocker())
        sys.path.insert(0, {str(REPO_ROOT)!r})
        script = sys.argv[1]
        sys.argv = [script, "--list"]
        runpy.run_path(script, run_name="__main__")
        """
    ).strip()

    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, str(REPO_ROOT / "scripts" / "release_contract_scan.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "SCAN 001 [artifact] built wheel exists" in completed.stdout


def test_help_paths_do_not_require_repo_helper_modules(tmp_path):
    scripts_and_blocked_modules = [
        ("scripts/full_scan.py", ("scripts.workflow_contracts",)),
        ("scripts/install_release_extras.py", ("scripts.release_artifact_metadata",)),
        ("scripts/interop_smoke.py", ("scripts.contracts",)),
        ("scripts/metadata_consistency.py", ("scripts.contracts",)),
        ("scripts/public_surface_scan.py", ("scripts.contracts",)),
        ("scripts/release_contract_scan.py", ("scripts.contracts", "scripts.release_artifact_metadata")),
        ("scripts/release_smoke.py", ("scripts.contracts",)),
    ]

    for relative_script, blocked_modules in scripts_and_blocked_modules:
        blocked_literals = ", ".join(repr(name) for name in blocked_modules)
        bootstrap = textwrap.dedent(
            f"""
            import importlib.abc
            import runpy
            import sys

            BLOCKED = ({blocked_literals},)

            class _HelperBlocker(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if any(fullname == blocked or fullname.startswith(blocked + ".") for blocked in BLOCKED):
                        raise ModuleNotFoundError(f"blocked import: {{fullname}}")
                    return None

            sys.meta_path.insert(0, _HelperBlocker())
            sys.path.insert(0, {str(REPO_ROOT)!r})
            script = sys.argv[1]
            sys.argv = [script, "--help"]
            runpy.run_path(script, run_name="__main__")
            """
        ).strip()

        completed = subprocess.run(
            [sys.executable, "-c", bootstrap, str(REPO_ROOT / relative_script)],
            cwd=tmp_path,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stderr


def test_interop_smoke_list_backends_does_not_require_repo_contracts(tmp_path):
    bootstrap = textwrap.dedent(
        f"""
        import importlib.abc
        import runpy
        import sys

        class _ContractsBlocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "scripts.contracts" or fullname.startswith("scripts.contracts."):
                    raise ModuleNotFoundError(f"blocked import: {{fullname}}")
                return None

        sys.meta_path.insert(0, _ContractsBlocker())
        sys.path.insert(0, {str(REPO_ROOT)!r})
        script = sys.argv[1]
        sys.argv = [script, "--list-backends"]
        runpy.run_path(script, run_name="__main__")
        """
    ).strip()

    completed = subprocess.run(
        [sys.executable, "-c", bootstrap, str(REPO_ROOT / "scripts" / "interop_smoke.py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines() == ["pyg", "dgl"]


def test_benchmark_hotpaths_writes_json_outside_repo_root(tmp_path):
    output = tmp_path / "benchmarks" / "outside-repo.json"
    completed = _run_script_outside_repo(
        tmp_path,
        "scripts/benchmark_hotpaths.py",
        "--num-nodes",
        "100",
        "--num-edges",
        "500",
        "--num-queries",
        "25",
        "--num-partitions",
        "2",
        "--warmup",
        "1",
        "--repeats",
        "1",
        "--seed",
        "0",
        "--output",
        str(output),
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == "vgl_hotpaths"
    assert payload["schema_version"] == 1
    assert payload["metric_unit"] == "seconds"
    assert payload["generated_at_utc"].endswith("Z")
    assert payload["config"]["num_nodes"] == 100
    assert payload["runner"]["python_version"]
