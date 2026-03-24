from dataclasses import dataclass, field

import torch
from typing import Any, Callable

from vgl.dataloading.plan import PlanStage, SamplingPlan

StageHandler = Callable[[PlanStage, "MaterializationContext"], "MaterializationContext"]


def _resolve_feature_store(feature_store, graph):
    if feature_store is not None:
        return feature_store
    return getattr(graph, "feature_store", None)


def _induced_edge_ids(graph, node_ids: torch.Tensor) -> torch.Tensor:
    node_ids = torch.as_tensor(node_ids, dtype=torch.long).view(-1)
    edge_index = graph.edge_index
    node_mask = torch.zeros(graph.x.size(0), dtype=torch.bool, device=edge_index.device)
    if node_ids.numel() > 0:
        node_mask[node_ids] = True
    edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    return torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)[edge_mask]


def _induced_edge_ids_by_type(graph, node_ids_by_type: dict[str, torch.Tensor]) -> dict[tuple[str, str, str], torch.Tensor]:
    edge_ids_by_type = {}
    for edge_type, store in graph.edges.items():
        src_type, _, dst_type = edge_type
        edge_index = store.edge_index
        src_ids = torch.as_tensor(node_ids_by_type[src_type], dtype=torch.long).view(-1)
        dst_ids = torch.as_tensor(node_ids_by_type[dst_type], dtype=torch.long).view(-1)
        src_mask = torch.isin(edge_index[0], src_ids) if src_ids.numel() > 0 else torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        dst_mask = torch.isin(edge_index[1], dst_ids) if dst_ids.numel() > 0 else torch.zeros(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
        edge_mask = src_mask & dst_mask
        edge_ids_by_type[edge_type] = torch.arange(edge_index.size(1), dtype=torch.long, device=edge_index.device)[edge_mask]
    return edge_ids_by_type


def _resolve_stage_index(stage: "PlanStage", context: "MaterializationContext", *, type_name) -> Any:
    index = context.state[stage.params["index_key"]]
    if isinstance(index, dict):
        try:
            return index[type_name]
        except KeyError as exc:
            raise KeyError(f"missing staged index for {type_name!r}") from exc
    return index


def _record_materialized_features(context: "MaterializationContext", *, entity_kind: str, type_name, fetched: dict[str, Any]) -> None:
    state_key = "_materialized_node_features" if entity_kind == "node" else "_materialized_edge_features"
    materialized = context.state.setdefault(state_key, {})
    type_bucket = materialized.setdefault(type_name, {})
    type_bucket.update(fetched)


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
            feature_store=_resolve_feature_store(feature_store, graph),
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
            context.state["edge_ids_by_type"] = _induced_edge_ids_by_type(context.graph, expanded)
        else:
            context.state["node_ids"] = expanded
            context.state["edge_ids"] = _induced_edge_ids(context.graph, expanded)
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
        source = context.feature_store
        if source is None:
            raise ValueError("feature_store is required for feature fetch stages")
        output_key = stage.params["output_key"]
        type_name = stage.params[type_key]
        if entity_kind == "edge":
            type_name = tuple(type_name)
        index = _resolve_stage_index(stage, context, type_name=type_name)

        direct_fetch = getattr(source, "fetch", None)
        node_fetch = getattr(source, "fetch_node_features", None)
        edge_fetch = getattr(source, "fetch_edge_features", None)

        fetched = {}
        for feature_name in stage.params["feature_names"]:
            key = (entity_kind, type_name, feature_name)
            if callable(direct_fetch):
                fetched[feature_name] = direct_fetch(key, index)
            elif entity_kind == "node" and callable(node_fetch):
                fetched[feature_name] = node_fetch(key, index)
            elif entity_kind == "edge" and callable(edge_fetch):
                fetched[feature_name] = edge_fetch(key, index)
            else:
                raise TypeError(
                    f"feature_store does not support {entity_kind} feature fetch for key {key!r}"
                )
        context.state[output_key] = fetched
        _record_materialized_features(context, entity_kind=entity_kind, type_name=type_name, fetched=fetched)
        return context
