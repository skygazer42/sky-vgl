import torch
from torch import nn

from gnn import Graph
from gnn.core.batch import TemporalEventBatch
from gnn.data.sample import TemporalEventRecord
from gnn.train.tasks import TemporalEventPredictionTask
from gnn.train.trainer import Trainer


def _batch():
    graph = Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "timestamp": torch.tensor([1, 4]),
            }
        },
        time_attr="timestamp",
    )
    return TemporalEventBatch.from_records(
        [
            TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=1, label=1),
            TemporalEventRecord(graph=graph, src_index=1, dst_index=2, timestamp=4, label=0),
        ]
    )


class TinyTemporalEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(9, 2)

    def forward(self, batch):
        src_x = batch.graph.x[batch.src_index]
        dst_x = batch.graph.x[batch.dst_index]
        history_counts = torch.tensor(
            [batch.history_graph(i).edge_index.size(1) for i in range(batch.labels.size(0))],
            dtype=src_x.dtype,
        ).unsqueeze(-1)
        return self.linear(torch.cat([src_x, dst_x, history_counts], dim=-1))


def test_trainer_runs_temporal_event_prediction_epoch():
    batch = _batch()
    trainer = Trainer(
        model=TinyTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit([batch])

    assert history["epochs"] == 1
