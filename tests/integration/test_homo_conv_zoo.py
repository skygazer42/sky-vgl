import torch
from torch import nn

from vgl import (
    AGNNConv,
    APPNPConv,
    ARMAConv,
    AntiSymmetricConv,
    BernConv,
    CGConv,
    ChebConv,
    ClusterGCNConv,
    DAGNNConv,
    DNAConv,
    DirGNNConv,
    EdgeConv,
    EGConv,
    FAConv,
    FAGCNConv,
    FiLMConv,
    FeaStConv,
    GATConv,
    GeneralConv,
    GATv2Conv,
    GCN2Conv,
    GatedGCNConv,
    GatedGraphConv,
    GENConv,
    GINEConv,
    GINConv,
    GMMConv,
    GPRGNNConv,
    GPSLayer,
    Graph,
    GraphConv,
    GraphTransformerEncoder,
    GraphormerEncoder,
    GroupRevRes,
    H2GCNConv,
    LEConv,
    LGConv,
    LightGCNConv,
    MFConv,
    MixHopConv,
    NNConv,
    NodeClassificationTask,
    PNAConv,
    PDNConv,
    RGCNConv,
    ResGatedGraphConv,
    SGConv,
    SAGEConv,
    SGFormerEncoder,
    SSGConv,
    SimpleConv,
    SplineConv,
    PointNetConv,
    PointTransformerConv,
    SuperGATConv,
    TAGConv,
    TransformerConv,
    Trainer,
    TWIRLSConv,
    NAGphormerEncoder,
    WLConvContinuous,
)


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
        pos=torch.randn(3, 2),
        edge_data={"edge_attr": torch.randn(3, 2), "pseudo": torch.rand(3, 2)},
        y=torch.tensor([0, 1, 0]),
        train_mask=torch.tensor([True, True, True]),
        val_mask=torch.tensor([True, True, True]),
        test_mask=torch.tensor([True, True, True]),
    )


def _model(conv):
    class TinyModel(nn.Module):
        def __init__(self, op):
            super().__init__()
            self.op = op
            out_channels = getattr(op, "out_channels", None)
            if out_channels is None:
                out_channels = getattr(op, "channels", 4)
            heads = getattr(op, "heads", 1)
            concat = getattr(op, "concat", False)
            hidden = out_channels * heads if concat else out_channels
            self.head = nn.Linear(hidden, 2)

        def forward(self, graph):
            if isinstance(self.op, GCN2Conv):
                return self.head(self.op(graph, x0=graph.x))
            if isinstance(self.op, FAConv):
                return self.head(self.op(graph, x0=graph.x))
            if isinstance(self.op, DNAConv):
                history = torch.stack([graph.x, graph.x * 0.5, graph.x + 0.25], dim=1)
                return self.head(self.op(graph, history=history))
            return self.head(self.op(graph))

    return TinyModel(conv)


def test_new_homo_convs_plug_into_training_loop():
    convs = [
        GINConv(in_channels=4, out_channels=4),
        GATConv(in_channels=4, out_channels=4),
        GATv2Conv(in_channels=4, out_channels=4, heads=2, concat=False),
        SAGEConv(in_channels=4, out_channels=4),
        APPNPConv(in_channels=4, out_channels=4, steps=2, alpha=0.1),
        TAGConv(in_channels=4, out_channels=4, k=2),
        SGConv(in_channels=4, out_channels=4, k=2),
        ChebConv(in_channels=4, out_channels=4, k=3),
        AGNNConv(channels=4),
        LightGCNConv(),
        LGConv(),
        FAGCNConv(channels=4, eps=0.1),
        ARMAConv(channels=4, stacks=2, layers=2, alpha=0.1),
        GPRGNNConv(channels=4, steps=3, alpha=0.1),
        MixHopConv(in_channels=4, out_channels=4, powers=(0, 1, 2)),
        BernConv(channels=4, steps=3),
        CGConv(channels=4, edge_channels=2, aggr="mean"),
        DNAConv(channels=4, heads=2),
        SSGConv(channels=4, steps=3, alpha=0.1),
        DAGNNConv(channels=4, steps=3),
        GCN2Conv(channels=4, alpha=0.1, theta=1.0, layer=1),
        GraphConv(in_channels=4, out_channels=4),
        H2GCNConv(in_channels=4, out_channels=4),
        EGConv(in_channels=4, out_channels=4, aggregators=("sum", "mean", "max")),
        GatedGCNConv(in_channels=4, out_channels=4, edge_channels=2),
        LEConv(in_channels=4, out_channels=4),
        ResGatedGraphConv(in_channels=4, out_channels=4),
        GatedGraphConv(channels=4, steps=2),
        ClusterGCNConv(in_channels=4, out_channels=4, diag_lambda=0.0),
        GENConv(in_channels=4, out_channels=4, aggr="softmax", beta=1.0),
        GINEConv(in_channels=4, out_channels=4, edge_channels=2),
        FiLMConv(in_channels=4, out_channels=4),
        FAConv(channels=4, eps=0.1),
        SimpleConv(aggr="mean"),
        EdgeConv(in_channels=4, out_channels=4, aggr="max"),
        NNConv(in_channels=4, out_channels=4, edge_channels=2),
        FeaStConv(in_channels=4, out_channels=4, heads=2),
        MFConv(in_channels=4, out_channels=4, max_degree=4),
        GMMConv(in_channels=4, out_channels=4, dim=2, kernel_size=3),
        PDNConv(in_channels=4, out_channels=4, edge_channels=2, add_self_loops=False),
        PointNetConv(in_channels=4, out_channels=4, pos_channels=2),
        PointTransformerConv(in_channels=4, out_channels=4, pos_channels=2),
        SplineConv(in_channels=4, out_channels=4, dim=2, kernel_size=3),
        PNAConv(
            in_channels=4,
            out_channels=4,
            aggregators=("sum", "mean", "max"),
            scalers=("identity", "amplification", "attenuation"),
        ),
        GeneralConv(
            in_channels=4,
            out_channels=4,
            aggr="add",
            heads=2,
            attention=True,
        ),
        AntiSymmetricConv(channels=4, num_iters=2, epsilon=0.1, gamma=0.1),
        TransformerConv(in_channels=4, out_channels=4, heads=2, concat=False, beta=True),
        GraphTransformerEncoder(channels=4, num_layers=2, heads=2),
        GraphormerEncoder(channels=4, num_layers=2, heads=2),
        GPSLayer(channels=4, local_gnn=GraphConv(in_channels=4, out_channels=4), heads=2),
        NAGphormerEncoder(channels=4, num_layers=2, num_hops=2, heads=2),
        SGFormerEncoder(channels=4, num_layers=2, heads=2, alpha=0.5),
        TWIRLSConv(in_channels=4, out_channels=4, steps=3, alpha=0.2),
        WLConvContinuous(),
        SuperGATConv(in_channels=4, out_channels=4, heads=2, concat=False),
        DirGNNConv(GraphConv(in_channels=4, out_channels=4), alpha=0.5, root_weight=True),
        GroupRevRes(LGConv(), num_groups=2),
    ]

    for conv in convs:
        trainer = Trainer(
            model=_model(conv),
            task=NodeClassificationTask(
                target="y",
                split=("train_mask", "val_mask", "test_mask"),
                metrics=["accuracy"],
            ),
            optimizer=torch.optim.Adam,
            lr=1e-2,
            max_epochs=1,
        )
        history = trainer.fit(_graph(), val_data=_graph())

        assert history["epochs"] == 1
        assert "loss" in history["train"][0]


def test_rgcn_conv_plugs_into_hetero_training_loop():
    class TinyHeteroModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = RGCNConv(
                in_channels=4,
                out_channels=4,
                node_types=("author", "paper"),
                relation_types=(
                    ("author", "writes", "paper"),
                    ("paper", "written_by", "author"),
                ),
            )
            self.head = nn.Linear(4, 2)

        def forward(self, graph):
            out = self.conv(graph)
            return self.head(out["paper"])

    graph = Graph.hetero(
        nodes={
            "paper": {
                "x": torch.randn(2, 4),
                "y": torch.tensor([0, 1]),
                "train_mask": torch.tensor([True, True]),
                "val_mask": torch.tensor([True, True]),
                "test_mask": torch.tensor([True, True]),
            },
            "author": {"x": torch.randn(2, 4)},
        },
        edges={
            ("author", "writes", "paper"): {"edge_index": torch.tensor([[0, 1], [1, 0]])},
            ("paper", "written_by", "author"): {"edge_index": torch.tensor([[1, 0], [0, 1]])},
        },
    )
    trainer = Trainer(
        model=TinyHeteroModel(),
        task=NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            node_type="paper",
            metrics=["accuracy"],
        ),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(graph, val_data=graph)

    assert history["epochs"] == 1
    assert "loss" in history["train"][0]
