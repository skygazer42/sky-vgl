import torch
from torch import nn

from vgl import (
    APPNPConv,
    ChebConv,
    GATv2Conv,
    GINConv,
    Graph,
    NodeClassificationTask,
    SGConv,
    TAGConv,
    Trainer,
)


def _graph():
    return Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.randn(3, 4),
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
            out_channels = getattr(op, "out_channels", 2)
            heads = getattr(op, "heads", 1)
            concat = getattr(op, "concat", False)
            hidden = out_channels * heads if concat else out_channels
            self.head = nn.Linear(hidden, 2)

        def forward(self, graph):
            return self.head(self.op(graph))

    return TinyModel(conv)


def test_new_homo_convs_plug_into_training_loop():
    convs = [
        GINConv(in_channels=4, out_channels=4),
        GATv2Conv(in_channels=4, out_channels=4, heads=2, concat=False),
        APPNPConv(in_channels=4, out_channels=4, steps=2, alpha=0.1),
        TAGConv(in_channels=4, out_channels=4, k=2),
        SGConv(in_channels=4, out_channels=4, k=2),
        ChebConv(in_channels=4, out_channels=4, k=3),
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
