import torch
from torch import nn

from vgl import Graph
from vgl.core.batch import LinkPredictionBatch
from vgl.data.sample import LinkPredictionRecord
from vgl.train.tasks import LinkPredictionTask
from vgl.train.trainer import Trainer


def _batch():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.randn(3, 4),
    )
    return LinkPredictionBatch.from_records(
        [
            LinkPredictionRecord(graph=graph, src_index=0, dst_index=1, label=1),
            LinkPredictionRecord(graph=graph, src_index=2, dst_index=0, label=0),
        ]
    )


class TinyLinkPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(4, 4)
        self.scorer = nn.Linear(8, 1)

    def forward(self, batch):
        node_repr = self.encoder(batch.graph.x)
        src_x = node_repr[batch.src_index]
        dst_x = node_repr[batch.dst_index]
        return self.scorer(torch.cat([src_x, dst_x], dim=-1)).squeeze(-1)


def test_trainer_runs_link_prediction_epoch():
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit([_batch()])

    assert history["epochs"] == 1
