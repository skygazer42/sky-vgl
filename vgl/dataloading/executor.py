from dataclasses import dataclass, field
from typing import Any, Callable

from vgl.dataloading.plan import PlanStage, SamplingPlan

StageHandler = Callable[[PlanStage, "MaterializationContext"], "MaterializationContext"]


@dataclass(slots=True)
class MaterializationContext:
    request: Any
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    graph: Any | None = None
    feature_store: Any | None = None


@dataclass(slots=True)
class PlanExecutor:
    handlers: dict[str, StageHandler] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.handlers.setdefault("expand_neighbors", self._expand_neighbors)
        self.handlers.setdefault("fetch_node_features", self._fetch_node_features)
        self.handlers.setdefault("fetch_edge_features", self._fetch_edge_features)
        self.handlers.setdefault("sample_link_neighbors", self._sample_link_neighbors)
        self.handlers.setdefault("sample_temporal_neighbors", self._sample_temporal_neighbors)

    def register(self, name: str, handler: StageHandler) -> None:
        self.handlers[name] = handler

    def execute(
        self,
        plan: SamplingPlan,
        *,
        graph: Any | None = None,
        feature_store: Any | None = None,
        state: dict[str, Any] | None = None,
    ) -> MaterializationContext:
        context = MaterializationContext(
            request=plan.request,
            state=dict(state or {}),
            metadata=dict(plan.metadata),
            graph=graph,
            feature_store=feature_store,
        )
        for stage in plan.stages:
            try:
                handler = self.handlers[stage.name]
            except KeyError as exc:
                raise KeyError(f"unknown stage handler: {stage.name}") from exc
            context = handler(stage, context)
        return context

    def _expand_neighbors(self, stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        from vgl.ops.khop import expand_neighbors

        if context.graph is None:
            raise ValueError("graph is required for neighbor expansion stages")
        request = context.request
        expanded = expand_neighbors(
            context.graph,
            request.node_ids,
            num_neighbors=stage.params["num_neighbors"],
            node_type=stage.params.get("node_type", getattr(request, "node_type", None)),
        )
        if isinstance(expanded, dict):
            context.state["node_ids_by_type"] = expanded
        else:
            context.state["node_ids"] = expanded
        return context

    def _fetch_node_features(self, stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        return self._fetch_features(stage, context, entity_kind="node", type_key="node_type")

    def _fetch_edge_features(self, stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        return self._fetch_features(stage, context, entity_kind="edge", type_key="edge_type")

    @staticmethod
    def _sample_link_neighbors(stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        sampler = stage.params["sampler"]
        records = list(stage.params["records"])
        sampled = sampler._sample_from_seed_records(records, is_sequence=bool(stage.params["is_sequence"]))
        if isinstance(sampled, (list, tuple)):
            context.state["records"] = list(sampled)
        else:
            context.state["record"] = sampled
        return context

    @staticmethod
    def _sample_temporal_neighbors(stage: PlanStage, context: MaterializationContext) -> MaterializationContext:
        sampler = stage.params["sampler"]
        context.state["record"] = sampler._sample_event(stage.params["record"])
        return context

    @staticmethod
    def _fetch_features(
        stage: PlanStage,
        context: MaterializationContext,
        *,
        entity_kind: str,
        type_key: str,
    ) -> MaterializationContext:
        if context.feature_store is None:
            raise ValueError("feature_store is required for feature fetch stages")
        index = context.state[stage.params["index_key"]]
        output_key = stage.params["output_key"]
        type_name = stage.params[type_key]
        fetched = {}
        for feature_name in stage.params["feature_names"]:
            key = (entity_kind, type_name, feature_name)
            fetched[feature_name] = context.feature_store.fetch(key, index)
        context.state[output_key] = fetched
        return context
