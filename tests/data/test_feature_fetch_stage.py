import torch

from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.requests import NodeSeedRequest
from vgl.storage import FeatureStore, InMemoryTensorStore


EDGE_TYPE = ("paper", "cites", "paper")


def test_executor_fetches_node_and_edge_features_from_feature_store():
    feature_store = FeatureStore(
        {
            ("node", "paper", "x"): InMemoryTensorStore(torch.tensor([[1.0], [2.0], [3.0]])),
            ("edge", EDGE_TYPE, "weight"): InMemoryTensorStore(torch.tensor([0.1, 0.2, 0.3, 0.4])),
        }
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([2, 0]), node_type="paper"),
        stages=(
            PlanStage(
                "fetch_node_features",
                params={
                    "node_type": "paper",
                    "feature_names": ("x",),
                    "index_key": "node_ids",
                    "output_key": "node_features",
                },
            ),
            PlanStage(
                "fetch_edge_features",
                params={
                    "edge_type": EDGE_TYPE,
                    "feature_names": ("weight",),
                    "index_key": "edge_ids",
                    "output_key": "edge_features",
                },
            ),
        ),
    )

    context = PlanExecutor().execute(
        plan,
        feature_store=feature_store,
        state={
            "node_ids": torch.tensor([2, 0]),
            "edge_ids": torch.tensor([3, 1]),
        },
    )

    node_slice = context.state["node_features"]["x"]
    edge_slice = context.state["edge_features"]["weight"]

    assert node_slice.index.tolist() == [2, 0]
    assert torch.equal(node_slice.values, torch.tensor([[3.0], [1.0]]))
    assert edge_slice.index.tolist() == [3, 1]
    assert torch.equal(edge_slice.values, torch.tensor([0.4, 0.2]))
