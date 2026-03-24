import torch

from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.requests import NodeSeedRequest


def test_sampling_plan_preserves_stage_order_and_metadata():
    request = NodeSeedRequest(
        node_ids=torch.tensor([2, 0]),
        node_type="paper",
        metadata={"split": "train"},
    )
    plan = SamplingPlan(
        request=request,
        stages=(
            PlanStage("expand_neighbors", params={"num_hops": 2}),
            PlanStage("fetch_features", params={"feature_names": ("x",)}),
        ),
        metadata={"sampler": "node"},
    )

    appended = plan.append(PlanStage("materialize"))

    assert [stage.name for stage in plan.stages] == ["expand_neighbors", "fetch_features"]
    assert [stage.name for stage in appended.stages] == ["expand_neighbors", "fetch_features", "materialize"]
    assert plan.metadata == {"sampler": "node"}
    assert appended.request is request
