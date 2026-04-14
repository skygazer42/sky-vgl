import torch

from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.requests import NodeSeedRequest


def test_plan_executor_runs_registered_stages_in_order():
    request = NodeSeedRequest(node_ids=torch.tensor([0, 1]), node_type="paper")
    plan = SamplingPlan(
        request=request,
        stages=(
            PlanStage("seed", params={"value": 1}),
            PlanStage("bump", params={"delta": 2}),
        ),
        metadata={"job": "demo"},
    )
    executor = PlanExecutor()

    def seed(stage, context):
        context.state["order"].append(stage.name)
        context.state["value"] = stage.params["value"]
        return context

    def bump(stage, context):
        context.state["order"].append(stage.name)
        context.state["value"] += stage.params["delta"]
        return context

    executor.register("seed", seed)
    executor.register("bump", bump)
    context = executor.execute(plan, state={"order": []})

    assert context.request is request
    assert context.metadata["job"] == "demo"
    assert context.metadata["stage_count"] == 2
    assert context.state["order"] == ["seed", "bump"]
    assert context.state["value"] == 3


def test_plan_executor_records_stage_level_profile_metadata():
    request = NodeSeedRequest(node_ids=torch.tensor([0]), node_type="paper")
    plan = SamplingPlan(
        request=request,
        stages=(
            PlanStage("seed", params={"value": 1}),
            PlanStage("bump", params={"delta": 2}),
        ),
        metadata={"job": "profiled"},
    )
    executor = PlanExecutor()

    def seed(stage, context):
        context.state["value"] = stage.params["value"]
        return context

    def bump(stage, context):
        context.state["value"] += stage.params["delta"]
        return context

    executor.register("seed", seed)
    executor.register("bump", bump)
    context = executor.execute(plan)

    assert context.metadata["job"] == "profiled"
    assert context.metadata["stage_count"] == 2
    assert context.metadata["stage_seconds_total"] >= 0.0
    assert [entry["stage_index"] for entry in context.metadata["stage_profile"]] == [0, 1]
    assert [entry["stage_name"] for entry in context.metadata["stage_profile"]] == ["seed", "bump"]
    assert all(entry["duration_seconds"] >= 0.0 for entry in context.metadata["stage_profile"])


def test_plan_executor_appends_stage_profile_after_existing_plan_metadata():
    request = NodeSeedRequest(node_ids=torch.tensor([0]), node_type="paper")
    plan = SamplingPlan(
        request=request,
        stages=(PlanStage("seed"),),
        metadata={"job": "demo", "owner": "lane-b"},
    )
    executor = PlanExecutor()
    executor.register("seed", lambda stage, context: context)

    context = executor.execute(plan, state={"order": []})

    assert context.metadata["job"] == "demo"
    assert context.metadata["owner"] == "lane-b"
    assert context.metadata["stage_count"] == 1
    assert context.metadata["stage_profile"][0]["stage_name"] == "seed"
