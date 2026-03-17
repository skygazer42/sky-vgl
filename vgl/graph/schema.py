from dataclasses import dataclass

from vgl.graph.errors import SchemaError


@dataclass(frozen=True, slots=True)
class GraphSchema:
    node_types: tuple[str, ...]
    edge_types: tuple[tuple[str, str, str], ...]
    node_features: dict[str, tuple[str, ...]]
    edge_features: dict[tuple[str, str, str], tuple[str, ...]]
    time_attr: str | None = None

    def __post_init__(self) -> None:
        if self.time_attr is None:
            return
        if any(self.time_attr in fields for fields in self.node_features.values()):
            return
        if any(self.time_attr in fields for fields in self.edge_features.values()):
            return
        raise SchemaError(f"time_attr '{self.time_attr}' is not declared in schema")
