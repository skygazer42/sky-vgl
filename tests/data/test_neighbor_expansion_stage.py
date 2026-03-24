import torch

from vgl import Graph
from vgl.dataloading.executor import PlanExecutor
from vgl.dataloading.plan import PlanStage, SamplingPlan
from vgl.dataloading.requests import NodeSeedRequest


def test_executor_expands_homo_node_seeds_across_neighbor_stages():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 3], [1, 2, 1]]),
        x=torch.arange(12, dtype=torch.float32).view(4, 3),
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([1]), node_type="node"),
        stages=(PlanStage("expand_neighbors", params={"num_neighbors": (-1,)}),),
    )

    context = PlanExecutor().execute(plan, graph=graph)

    assert torch.equal(context.state["node_ids"], torch.tensor([0, 1, 2, 3]))
    assert torch.equal(context.state["edge_ids"], torch.tensor([0, 1, 2]))


def test_executor_expands_hetero_node_seeds_by_type():
    graph = Graph.hetero(
        nodes={
            "paper": {"x": torch.randn(3, 4)},
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]])
            },
            ("paper", "written_by", "author"): {
                "edge_index": torch.tensor([[1, 2], [0, 1]])
            },
        },
    )
    plan = SamplingPlan(
        request=NodeSeedRequest(node_ids=torch.tensor([1]), node_type="paper"),
        stages=(PlanStage("expand_neighbors", params={"num_neighbors": (-1,)}),),
    )

    context = PlanExecutor().execute(plan, graph=graph)

    assert torch.equal(context.state["node_ids_by_type"]["paper"], torch.tensor([1]))
    assert torch.equal(context.state["node_ids_by_type"]["author"], torch.tensor([0]))
    assert torch.equal(
        context.state["edge_ids_by_type"][("author", "writes", "paper")],
        torch.tensor([0]),
    )
    assert torch.equal(
        context.state["edge_ids_by_type"][("paper", "written_by", "author")],
        torch.tensor([0]),
    )
