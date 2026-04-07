from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


def _read_release_version() -> str:
    version_path = Path(__file__).resolve().parents[1] / "vgl" / "version.py"
    match = re.search(
        r"""__version__\s*=\s*["']([^"']+)["']""",
        version_path.read_text(encoding="utf-8"),
    )
    if match is None:
        raise RuntimeError("unable to parse __version__ from vgl/version.py")
    return match.group(1)


RELEASE_VERSION = _read_release_version()
PROJECT_NAME = "sky-vgl"
REQUIRES_PYTHON = ">=3.10"
MIN_TORCH_VERSION = "2.4"
DOCS_SITE_URL = "https://skygazer42.github.io/sky-vgl"
PROJECT_URLS = {
    "Homepage": "https://github.com/skygazer42/sky-vgl",
    "Repository": "https://github.com/skygazer42/sky-vgl",
    "Documentation": DOCS_SITE_URL,
    "Issues": "https://github.com/skygazer42/sky-vgl/issues",
    "Changelog": "https://github.com/skygazer42/sky-vgl/releases",
}
OPTIONAL_EXTRAS = ("dev", "scipy", "networkx", "tensorboard", "dgl", "pyg", "full")
DEFAULT_LIGHTWEIGHT_EXTRAS = ("networkx", "scipy", "tensorboard")
REAL_INTEROP_BACKENDS = ("pyg", "dgl")
INTEROP_SMOKE_SCRIPT = "scripts/interop_smoke.py"
SUPPORTED_PYTHON_VERSIONS = ("3.10", "3.11", "3.12")
PRIMARY_CI_PLATFORMS = ("ubuntu-latest", "macos-latest")
PRIMARY_MACOS_SMOKE = "macos-latest"
WHEEL_REQUIRED_FILES = ("vgl/__init__.py", "vgl/version.py")
WHEEL_EXCLUDED_SUBSTRINGS = ("/.factory/", "/docs/plans/", "__pycache__", "examples/")
SDIST_REQUIRED_SUFFIXES = (
    "/README.md",
    "/LICENSE",
    "/docs/releasing.md",
    "/scripts/release_smoke.py",
    "/scripts/interop_smoke.py",
)
SDIST_EXCLUDED_SUBSTRINGS = ("/.factory/", "/docs/plans/", "__pycache__")
WHEEL_IMPORT_SYMBOLS = ("Graph", "Trainer", "PlanetoidDataset", "NodeClassificationTask")
README_VERSION_BADGE = "https://img.shields.io/pypi/v/sky-vgl?style=flat-square"
DOCS_INDEX_VERSION_BADGE = "https://img.shields.io/pypi/v/sky-vgl?style=for-the-badge"
PUBLIC_EXAMPLE_MODULES = (
    "examples/hetero/graph_classification.py",
    "examples/hetero/link_prediction.py",
    "examples/hetero/node_classification.py",
    "examples/homo/cluster_gcn_node_classification.py",
    "examples/homo/conv_zoo.py",
    "examples/homo/graph_classification.py",
    "examples/homo/graph_saint_node_classification.py",
    "examples/homo/link_prediction.py",
    "examples/homo/node_classification.py",
    "examples/homo/planetoid_node_classification.py",
    "examples/homo/tu_graph_classification.py",
    "examples/temporal/event_prediction.py",
    "examples/temporal/memory_event_prediction.py",
)
REPRESENTATIVE_EXAMPLE_SMOKES = (
    "examples/homo/node_classification.py",
    "examples/homo/graph_classification.py",
    "examples/hetero/node_classification.py",
    "examples/temporal/event_prediction.py",
)
FORBIDDEN_PREFERRED_IMPORT_PREFIXES = ("vgl.core", "vgl.data", "vgl.train")


@dataclass(frozen=True)
class ReexportSpec:
    category: str
    description: str
    relative_path: str
    symbol: str
    expected_source: str


@dataclass(frozen=True)
class LegacyNamespacePolicy:
    module: str
    preferred_module: str
    guidance: str


GOLDEN_PATH_ROOT_EXPORTS = (
    ReexportSpec("root", "vgl exports Graph from vgl.graph", "vgl/__init__.py", "Graph", "vgl.graph.Graph"),
    ReexportSpec(
        "root",
        "vgl exports GraphBatch from vgl.graph",
        "vgl/__init__.py",
        "GraphBatch",
        "vgl.graph.GraphBatch",
    ),
    ReexportSpec(
        "root",
        "vgl exports DataLoader from vgl.dataloading",
        "vgl/__init__.py",
        "DataLoader",
        "vgl.dataloading.DataLoader",
    ),
    ReexportSpec("root", "vgl exports Trainer from vgl.engine", "vgl/__init__.py", "Trainer", "vgl.engine.Trainer"),
    ReexportSpec(
        "root",
        "vgl exports MessagePassing from vgl.nn",
        "vgl/__init__.py",
        "MessagePassing",
        "vgl.nn.MessagePassing",
    ),
    ReexportSpec(
        "root",
        "vgl exports NodeClassificationTask from vgl.tasks",
        "vgl/__init__.py",
        "NodeClassificationTask",
        "vgl.tasks.NodeClassificationTask",
    ),
    ReexportSpec(
        "root",
        "vgl exports GraphClassificationTask from vgl.tasks",
        "vgl/__init__.py",
        "GraphClassificationTask",
        "vgl.tasks.GraphClassificationTask",
    ),
    ReexportSpec(
        "root",
        "vgl exports LinkPredictionTask from vgl.tasks",
        "vgl/__init__.py",
        "LinkPredictionTask",
        "vgl.tasks.LinkPredictionTask",
    ),
    ReexportSpec(
        "root",
        "vgl exports TemporalEventPredictionTask from vgl.tasks",
        "vgl/__init__.py",
        "TemporalEventPredictionTask",
        "vgl.tasks.TemporalEventPredictionTask",
    ),
    ReexportSpec(
        "root",
        "vgl exports __version__ from vgl.version",
        "vgl/__init__.py",
        "__version__",
        "vgl.version.__version__",
    ),
    ReexportSpec(
        "root",
        "vgl exports DatasetRegistry from vgl.data",
        "vgl/__init__.py",
        "DatasetRegistry",
        "vgl.data.DatasetRegistry",
    ),
    ReexportSpec(
        "root",
        "vgl exports KarateClubDataset from vgl.data",
        "vgl/__init__.py",
        "KarateClubDataset",
        "vgl.data.KarateClubDataset",
    ),
    ReexportSpec(
        "root",
        "vgl exports PlanetoidDataset from vgl.data",
        "vgl/__init__.py",
        "PlanetoidDataset",
        "vgl.data.PlanetoidDataset",
    ),
    ReexportSpec("root", "vgl exports TUDataset from vgl.data", "vgl/__init__.py", "TUDataset", "vgl.data.TUDataset"),
)

DATA_PUBLIC_EXPORTS = (
    ReexportSpec(
        "data",
        "vgl.data reexports DatasetRegistry from vgl.data.public",
        "vgl/data/__init__.py",
        "DatasetRegistry",
        "vgl.data.public.DatasetRegistry",
    ),
    ReexportSpec(
        "data",
        "vgl.data reexports KarateClubDataset from vgl.data.public",
        "vgl/data/__init__.py",
        "KarateClubDataset",
        "vgl.data.public.KarateClubDataset",
    ),
    ReexportSpec(
        "data",
        "vgl.data reexports PlanetoidDataset from vgl.data.public",
        "vgl/data/__init__.py",
        "PlanetoidDataset",
        "vgl.data.public.PlanetoidDataset",
    ),
    ReexportSpec(
        "data",
        "vgl.data reexports TUDataset from vgl.data.public",
        "vgl/data/__init__.py",
        "TUDataset",
        "vgl.data.public.TUDataset",
    ),
)

DATALOADING_PUBLIC_EXPORTS = (
    ReexportSpec(
        "dataloading",
        "vgl.dataloading reexports RandomWalkSampler from vgl.dataloading.advanced",
        "vgl/dataloading/__init__.py",
        "RandomWalkSampler",
        "vgl.dataloading.advanced.RandomWalkSampler",
    ),
    ReexportSpec(
        "dataloading",
        "vgl.dataloading reexports Node2VecWalkSampler from vgl.dataloading.advanced",
        "vgl/dataloading/__init__.py",
        "Node2VecWalkSampler",
        "vgl.dataloading.advanced.Node2VecWalkSampler",
    ),
    ReexportSpec(
        "dataloading",
        "vgl.dataloading reexports GraphSAINTNodeSampler from vgl.dataloading.advanced",
        "vgl/dataloading/__init__.py",
        "GraphSAINTNodeSampler",
        "vgl.dataloading.advanced.GraphSAINTNodeSampler",
    ),
    ReexportSpec(
        "dataloading",
        "vgl.dataloading reexports ClusterData from vgl.dataloading.advanced",
        "vgl/dataloading/__init__.py",
        "ClusterData",
        "vgl.dataloading.advanced.ClusterData",
    ),
    ReexportSpec(
        "dataloading",
        "vgl.dataloading reexports ClusterLoader from vgl.dataloading.advanced",
        "vgl/dataloading/__init__.py",
        "ClusterLoader",
        "vgl.dataloading.advanced.ClusterLoader",
    ),
    ReexportSpec(
        "dataloading",
        "vgl.dataloading reexports ShaDowKHopSampler from vgl.dataloading.advanced",
        "vgl/dataloading/__init__.py",
        "ShaDowKHopSampler",
        "vgl.dataloading.advanced.ShaDowKHopSampler",
    ),
    ReexportSpec(
        "dataloading",
        "vgl.dataloading reexports GraphSAINTEdgeSampler from vgl.dataloading.advanced",
        "vgl/dataloading/__init__.py",
        "GraphSAINTEdgeSampler",
        "vgl.dataloading.advanced.GraphSAINTEdgeSampler",
    ),
    ReexportSpec(
        "dataloading",
        "vgl.dataloading reexports GraphSAINTRandomWalkSampler from vgl.dataloading.advanced",
        "vgl/dataloading/__init__.py",
        "GraphSAINTRandomWalkSampler",
        "vgl.dataloading.advanced.GraphSAINTRandomWalkSampler",
    ),
)

LEGACY_REEXPORTS = (
    ReexportSpec(
        "legacy",
        "vgl.train reexports Evaluator from vgl.engine",
        "vgl/train/__init__.py",
        "Evaluator",
        "vgl.engine.Evaluator",
    ),
    ReexportSpec(
        "legacy",
        "vgl.train reexports TensorBoardLogger from vgl.engine",
        "vgl/train/__init__.py",
        "TensorBoardLogger",
        "vgl.engine.TensorBoardLogger",
    ),
    ReexportSpec(
        "legacy",
        "vgl.train reexports Trainer from vgl.engine",
        "vgl/train/__init__.py",
        "Trainer",
        "vgl.engine.Trainer",
    ),
    ReexportSpec(
        "legacy",
        "vgl.train reexports ModelCheckpoint from vgl.engine",
        "vgl/train/__init__.py",
        "ModelCheckpoint",
        "vgl.engine.ModelCheckpoint",
    ),
    ReexportSpec(
        "legacy",
        "vgl.train reexports HitsAtK from vgl.metrics",
        "vgl/train/__init__.py",
        "HitsAtK",
        "vgl.metrics.HitsAtK",
    ),
    ReexportSpec(
        "legacy",
        "vgl.train reexports FloodingTask from vgl.tasks",
        "vgl/train/__init__.py",
        "FloodingTask",
        "vgl.tasks.FloodingTask",
    ),
    ReexportSpec(
        "legacy",
        "vgl.train reexports SAM from vgl.engine",
        "vgl/train/__init__.py",
        "SAM",
        "vgl.engine.SAM",
    ),
    ReexportSpec("legacy", "vgl.core reexports Graph from vgl.graph", "vgl/core/__init__.py", "Graph", "vgl.graph.Graph"),
    ReexportSpec(
        "legacy",
        "vgl.core reexports EdgeStore from vgl.graph",
        "vgl/core/__init__.py",
        "EdgeStore",
        "vgl.graph.EdgeStore",
    ),
    ReexportSpec(
        "legacy",
        "vgl.core reexports NodeStore from vgl.graph",
        "vgl/core/__init__.py",
        "NodeStore",
        "vgl.graph.NodeStore",
    ),
    ReexportSpec(
        "legacy",
        "vgl.core reexports GNNError from vgl.graph",
        "vgl/core/__init__.py",
        "GNNError",
        "vgl.graph.GNNError",
    ),
    ReexportSpec(
        "legacy",
        "vgl.core reexports SchemaError from vgl.graph",
        "vgl/core/__init__.py",
        "SchemaError",
        "vgl.graph.SchemaError",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports LinkNeighborSampler from vgl.dataloading",
        "vgl/data/__init__.py",
        "LinkNeighborSampler",
        "vgl.dataloading.LinkNeighborSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports PlanStage from vgl.dataloading",
        "vgl/data/__init__.py",
        "PlanStage",
        "vgl.dataloading.PlanStage",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports SamplingPlan from vgl.dataloading",
        "vgl/data/__init__.py",
        "SamplingPlan",
        "vgl.dataloading.SamplingPlan",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports PlanExecutor from vgl.dataloading",
        "vgl/data/__init__.py",
        "PlanExecutor",
        "vgl.dataloading.PlanExecutor",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports MaterializationContext from vgl.dataloading",
        "vgl/data/__init__.py",
        "MaterializationContext",
        "vgl.dataloading.MaterializationContext",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports GraphSeedRequest from vgl.dataloading",
        "vgl/data/__init__.py",
        "GraphSeedRequest",
        "vgl.dataloading.GraphSeedRequest",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports NodeSeedRequest from vgl.dataloading",
        "vgl/data/__init__.py",
        "NodeSeedRequest",
        "vgl.dataloading.NodeSeedRequest",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports materialize_context from vgl.dataloading",
        "vgl/data/__init__.py",
        "materialize_context",
        "vgl.dataloading.materialize_context",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports materialize_batch from vgl.dataloading",
        "vgl/data/__init__.py",
        "materialize_batch",
        "vgl.dataloading.materialize_batch",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.plan reexports PlanStage from vgl.dataloading.plan",
        "vgl/data/plan.py",
        "PlanStage",
        "vgl.dataloading.plan.PlanStage",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.executor reexports PlanExecutor from vgl.dataloading.executor",
        "vgl/data/executor.py",
        "PlanExecutor",
        "vgl.dataloading.executor.PlanExecutor",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.requests reexports NodeSeedRequest from vgl.dataloading.requests",
        "vgl/data/requests.py",
        "NodeSeedRequest",
        "vgl.dataloading.requests.NodeSeedRequest",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.materialize reexports materialize_batch from vgl.dataloading.materialize",
        "vgl/data/materialize.py",
        "materialize_batch",
        "vgl.dataloading.materialize.materialize_batch",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.sampler reexports LinkNeighborSampler from vgl.dataloading.sampler",
        "vgl/data/sampler.py",
        "LinkNeighborSampler",
        "vgl.dataloading.sampler.LinkNeighborSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.sampler reexports RandomWalkSampler from vgl.dataloading.advanced",
        "vgl/data/sampler.py",
        "RandomWalkSampler",
        "vgl.dataloading.advanced.RandomWalkSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.sampler reexports GraphSAINTEdgeSampler from vgl.dataloading.advanced",
        "vgl/data/sampler.py",
        "GraphSAINTEdgeSampler",
        "vgl.dataloading.advanced.GraphSAINTEdgeSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.sampler reexports Node2VecWalkSampler from vgl.dataloading.advanced",
        "vgl/data/sampler.py",
        "Node2VecWalkSampler",
        "vgl.dataloading.advanced.Node2VecWalkSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.transform reexports BaseTransform from vgl.transforms",
        "vgl/data/transform.py",
        "BaseTransform",
        "vgl.transforms.BaseTransform",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.transform reexports NormalizeFeatures from vgl.transforms",
        "vgl/data/transform.py",
        "NormalizeFeatures",
        "vgl.transforms.NormalizeFeatures",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.transform reexports RandomNodeSplit from vgl.transforms",
        "vgl/data/transform.py",
        "RandomNodeSplit",
        "vgl.transforms.RandomNodeSplit",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data.transform reexports RandomLinkSplit from vgl.transforms",
        "vgl/data/transform.py",
        "RandomLinkSplit",
        "vgl.transforms.RandomLinkSplit",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports ClusterData from vgl.dataloading",
        "vgl/data/__init__.py",
        "ClusterData",
        "vgl.dataloading.ClusterData",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports ClusterLoader from vgl.dataloading",
        "vgl/data/__init__.py",
        "ClusterLoader",
        "vgl.dataloading.ClusterLoader",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports RandomWalkSampler from vgl.dataloading",
        "vgl/data/__init__.py",
        "RandomWalkSampler",
        "vgl.dataloading.RandomWalkSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports Node2VecWalkSampler from vgl.dataloading",
        "vgl/data/__init__.py",
        "Node2VecWalkSampler",
        "vgl.dataloading.Node2VecWalkSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports GraphSAINTNodeSampler from vgl.dataloading",
        "vgl/data/__init__.py",
        "GraphSAINTNodeSampler",
        "vgl.dataloading.GraphSAINTNodeSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports GraphSAINTEdgeSampler from vgl.dataloading",
        "vgl/data/__init__.py",
        "GraphSAINTEdgeSampler",
        "vgl.dataloading.GraphSAINTEdgeSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports GraphSAINTRandomWalkSampler from vgl.dataloading",
        "vgl/data/__init__.py",
        "GraphSAINTRandomWalkSampler",
        "vgl.dataloading.GraphSAINTRandomWalkSampler",
    ),
    ReexportSpec(
        "legacy",
        "vgl.data reexports ShaDowKHopSampler from vgl.dataloading",
        "vgl/data/__init__.py",
        "ShaDowKHopSampler",
        "vgl.dataloading.ShaDowKHopSampler",
    ),
)

LEGACY_NAMESPACE_POLICIES = (
    LegacyNamespacePolicy("vgl.train", "vgl.engine / vgl.tasks / vgl.metrics", "Legacy compatibility package."),
    LegacyNamespacePolicy("vgl.core", "vgl.graph", "Legacy compatibility package."),
    LegacyNamespacePolicy(
        "vgl.data",
        "vgl.data for datasets, vgl.dataloading for loaders and plans",
        "Mixed namespace kept for compatibility; prefer vgl.dataloading for sampling and loaders.",
    ),
)

SUPPORT_MATRIX = (
    ("Python", "3.10 / 3.11 / 3.12", "Declared and CI-verified target range."),
    ("PyTorch", "2.4+", "Core runtime dependency."),
    ("networkx extra", "CI lightweight smoke", "Interop adapter surface validated in CI."),
    ("scipy extra", "CI lightweight smoke", "Sparse export surface validated in CI."),
    ("tensorboard extra", "CI lightweight smoke", "Logger extra validated in CI."),
    ("pyg extra", "manual/nightly smoke", "Real-install smoke is exercised via scripts/interop_smoke.py."),
    ("dgl extra", "manual/nightly smoke", "Real-install smoke is exercised via scripts/interop_smoke.py."),
)


def public_surface_specs() -> tuple[ReexportSpec, ...]:
    return GOLDEN_PATH_ROOT_EXPORTS + DATA_PUBLIC_EXPORTS + DATALOADING_PUBLIC_EXPORTS + LEGACY_REEXPORTS
