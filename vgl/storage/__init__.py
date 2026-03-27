from vgl.storage.base import TensorSlice as TensorSlice
from vgl.storage.base import TensorStore as TensorStore
from vgl.storage.feature_store import FeatureStore as FeatureStore
from vgl.storage.graph_store import GraphStore as GraphStore
from vgl.storage.graph_store import InMemoryGraphStore as InMemoryGraphStore
from vgl.storage.memory import InMemoryTensorStore as InMemoryTensorStore

__all__ = [
    "TensorSlice",
    "TensorStore",
    "InMemoryTensorStore",
    "MmapTensorStore",
    "FeatureStore",
    "GraphStore",
    "InMemoryGraphStore",
]


def __getattr__(name: str):
    if name == "MmapTensorStore":
        from vgl.storage.mmap import MmapTensorStore

        return MmapTensorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
