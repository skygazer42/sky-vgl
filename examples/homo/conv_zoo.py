from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl import (
    AGNNConv,
    APPNPConv,
    ARMAConv,
    AntiSymmetricConv,
    BernConv,
    ChebConv,
    ClusterGCNConv,
    DAGNNConv,
    DirGNNConv,
    EdgeConv,
    EGConv,
    FAGCNConv,
    FiLMConv,
    FeaStConv,
    GeneralConv,
    GATv2Conv,
    GCN2Conv,
    GatedGCNConv,
    GatedGraphConv,
    GENConv,
    GINConv,
    GPRGNNConv,
    Graph,
    GraphConv,
    GroupRevRes,
    H2GCNConv,
    LEConv,
    LGConv,
    LightGCNConv,
    MFConv,
    MixHopConv,
    NodeClassificationTask,
    PDNConv,
    PointNetConv,
    PointTransformerConv,
    PNAConv,
    ResGatedGraphConv,
    SGConv,
    SSGConv,
    SimpleConv,
    SuperGATConv,
    TAGConv,
    TransformerConv,
    Trainer,
    WLConvContinuous,
)


def build_demo_graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
        pos=torch.randn(3, 2),
        edge_data={"edge_attr": torch.randn(3, 2)},
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.tensor([True, True, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )


class TinyConvModel(nn.Module):
    def __init__(self, conv, hidden_channels):
        super().__init__()
        self.conv = conv
        self.head = nn.Linear(hidden_channels, 2)

    def forward(self, graph):
        if isinstance(self.conv, GCN2Conv):
            return self.head(self.conv(graph, x0=graph.x))
        return self.head(self.conv(graph))


def run_one(name, conv, hidden_channels):
    graph = build_demo_graph()
    trainer = Trainer(
        model=TinyConvModel(conv, hidden_channels),
        task=NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            metrics=["accuracy"],
        ),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    history = trainer.fit(graph, val_data=graph)
    return {"name": name, "loss": history["train"][-1]["loss"]}


def main():
    results = [
        run_one("gin", GINConv(in_channels=4, out_channels=4), 4),
        run_one("gatv2", GATv2Conv(in_channels=4, out_channels=4, heads=2, concat=False), 4),
        run_one("appnp", APPNPConv(in_channels=4, out_channels=4, steps=2, alpha=0.1), 4),
        run_one("tag", TAGConv(in_channels=4, out_channels=4, k=2), 4),
        run_one("sg", SGConv(in_channels=4, out_channels=4, k=2), 4),
        run_one("cheb", ChebConv(in_channels=4, out_channels=4, k=3), 4),
        run_one("agnn", AGNNConv(channels=4), 4),
        run_one("lightgcn", LightGCNConv(), 4),
        run_one("lgconv", LGConv(), 4),
        run_one("fagcn", FAGCNConv(channels=4, eps=0.1), 4),
        run_one("arma", ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1), 4),
        run_one("gprgnn", GPRGNNConv(channels=4, steps=3, alpha=0.1), 4),
        run_one("mixhop", MixHopConv(in_channels=4, out_channels=4, powers=(0, 1, 2)), 4),
        run_one("bern", BernConv(channels=4, steps=3), 4),
        run_one("ssg", SSGConv(channels=4, steps=3, alpha=0.1), 4),
        run_one("dagnn", DAGNNConv(channels=4, steps=3), 4),
        run_one("gcn2", GCN2Conv(channels=4, alpha=0.1, theta=1.0, layer=1), 4),
        run_one("graphconv", GraphConv(in_channels=4, out_channels=4), 4),
        run_one("h2gcn", H2GCNConv(in_channels=4, out_channels=4), 4),
        run_one("egconv", EGConv(in_channels=4, out_channels=4, aggregators=("sum", "mean", "max")), 4),
        run_one("gatedgcn", GatedGCNConv(in_channels=4, out_channels=4, edge_channels=2), 4),
        run_one("leconv", LEConv(in_channels=4, out_channels=4), 4),
        run_one("resgated", ResGatedGraphConv(in_channels=4, out_channels=4), 4),
        run_one("gatedgraph", GatedGraphConv(channels=4, steps=2), 4),
        run_one("clustergcn", ClusterGCNConv(in_channels=4, out_channels=4, diag_lambda=0.0), 4),
        run_one("gen", GENConv(in_channels=4, out_channels=4, aggr="softmax", beta=1.0), 4),
        run_one("film", FiLMConv(in_channels=4, out_channels=4), 4),
        run_one("simple", SimpleConv(aggr="mean"), 4),
        run_one("edgeconv", EdgeConv(in_channels=4, out_channels=4, aggr="max"), 4),
        run_one("feast", FeaStConv(in_channels=4, out_channels=4, heads=2), 4),
        run_one("mfconv", MFConv(in_channels=4, out_channels=4, max_degree=4), 4),
        run_one("pdn", PDNConv(in_channels=4, out_channels=4, edge_channels=2, add_self_loops=False), 4),
        run_one("pointnet", PointNetConv(in_channels=4, out_channels=4, pos_channels=2), 4),
        run_one("pointtransformer", PointTransformerConv(in_channels=4, out_channels=4, pos_channels=2), 4),
        run_one(
            "pna",
            PNAConv(
                in_channels=4,
                out_channels=4,
                aggregators=("sum", "mean", "max"),
                scalers=("identity", "amplification", "attenuation"),
            ),
            4,
        ),
        run_one(
            "generalconv",
            GeneralConv(
                in_channels=4,
                out_channels=4,
                aggr="add",
                heads=2,
                attention=True,
            ),
            4,
        ),
        run_one("antisymmetric", AntiSymmetricConv(channels=4, num_iters=2, epsilon=0.1, gamma=0.1), 4),
        run_one(
            "transformerconv",
            TransformerConv(in_channels=4, out_channels=4, heads=2, concat=False, beta=True),
            4,
        ),
        run_one("wlconv", WLConvContinuous(), 4),
        run_one("supergat", SuperGATConv(in_channels=4, out_channels=4, heads=2, concat=False), 4),
        run_one("dirgnn", DirGNNConv(GraphConv(in_channels=4, out_channels=4), alpha=0.5, root_weight=True), 4),
        run_one("grouprevres", GroupRevRes(LGConv(), num_groups=2), 4),
    ]
    print(results)
    return results


if __name__ == "__main__":
    main()
