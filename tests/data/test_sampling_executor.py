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
    assert context.metadata == {"job": "demo"}
    assert context.state["order"] == ["seed", "bump"]
    assert context.state["value"] == 3
