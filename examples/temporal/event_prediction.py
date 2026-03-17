from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, FullGraphSampler, ListDataset, TemporalEventRecord
from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.nn import TGATEncoder, TimeEncoder
from vgl.tasks import TemporalEventPredictionTask


class TinyTemporalEventModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TGATEncoder(channels=4, num_layers=2, time_channels=4, heads=2, dropout=0.0)
        self.time_encoder = TimeEncoder(out_channels=4)
        self.linear = nn.Linear(12, 2)

    def forward(self, batch):
        features = []
        for index in range(batch.labels.size(0)):
            history = batch.history_graph(index)
            query_time = batch.timestamp[index].to(dtype=history.x.dtype)
            node_repr = self.encoder(history, query_time=query_time)
            src_x = node_repr[batch.src_index[index]]
            dst_x = node_repr[batch.dst_index[index]]
            time_x = self.time_encoder(query_time.unsqueeze(0)).squeeze(0)
            features.append(torch.cat([src_x, dst_x, time_x], dim=-1))
        return self.linear(torch.stack(features, dim=0))


def build_demo_graph():
    return Graph.temporal(
        nodes={"node": {"x": torch.randn(3, 4)}},
        edges={
            ("node", "interacts", "node"): {
                "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                "timestamp": torch.tensor([1, 3, 5]),
            }
        },
        time_attr="timestamp",
    )


def build_demo_loader():
    graph = build_demo_graph()
    samples = [
        TemporalEventRecord(graph=graph, src_index=0, dst_index=1, timestamp=3, label=1),
        TemporalEventRecord(graph=graph, src_index=2, dst_index=0, timestamp=5, label=0),
    ]
    return DataLoader(
        dataset=ListDataset(samples),
        sampler=FullGraphSampler(),
        batch_size=2,
    )


def main():
    trainer = Trainer(
        model=TinyTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )
    result = trainer.fit(build_demo_loader())
    print(result)
    return result


if __name__ == "__main__":
    main()
