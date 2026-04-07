import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

EXAMPLE_PATHS = [
    "examples/homo/node_classification.py",
    "examples/homo/graph_classification.py",
    "examples/homo/planetoid_node_classification.py",
    "examples/homo/graph_saint_node_classification.py",
    "examples/homo/cluster_gcn_node_classification.py",
    "examples/homo/link_prediction.py",
    "examples/homo/tu_graph_classification.py",
    "examples/homo/conv_zoo.py",
    "examples/hetero/node_classification.py",
    "examples/hetero/graph_classification.py",
    "examples/hetero/link_prediction.py",
    "examples/temporal/event_prediction.py",
    "examples/temporal/memory_event_prediction.py",
]

EXECUTION_PATHS = [
    "examples/homo/node_classification.py",
    "examples/homo/graph_classification.py",
    "examples/hetero/node_classification.py",
    "examples/temporal/event_prediction.py",
]


def _load_module(relative_path: str):
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_text(relative_path: str) -> str:
    return (REPO_ROOT / relative_path).read_text(encoding="utf-8")


def test_new_example_modules_import_cleanly():
    for relative_path in EXAMPLE_PATHS:
        module = _load_module(relative_path)
        assert hasattr(module, "main")


def test_examples_provide_main_guard():
    for relative_path in EXAMPLE_PATHS:
        text = _read_text(relative_path)
        assert (
            'if __name__ == "__main__":' in text
            or "if __name__ == '__main__':" in text
        ), f"missing __main__ guard in {relative_path}"


def test_representative_examples_run_quickly():
    for relative_path in EXECUTION_PATHS:
        module = _load_module(relative_path)
        result = module.main()
        assert result is None or isinstance(result, (dict, list))
