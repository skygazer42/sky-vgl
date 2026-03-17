from pathlib import Path
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vgl.dataloading import DataLoader, FullGraphSampler, ListDataset, TemporalEventRecord
from vgl.engine import Trainer
from vgl.graph import Graph
from vgl.nn import TGNMemory
from vgl.tasks import TemporalEventPredictionTask


class TinyTemporalMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory = TGNMemory(
            num_nodes=3,
            memory_channels=4,
            raw_message_channels=2,
            time_channels=4,
        )
        self.linear = nn.Linear(10, 2)

    def forward(self, batch):
        self.memory.reset_state()
        event_features = batch.event_features
        if event_features is None:
            event_features = self.linear.weight.new_zeros(batch.labels.size(0), self.memory.raw_message_channels)
        else:
            event_features = event_features.to(dtype=self.linear.weight.dtype, device=self.linear.weight.device)

        logits = self.linear.weight.new_zeros(batch.labels.size(0), 2)
        for index in torch.argsort(batch.timestamp).tolist():
            src_index = batch.src_index[index : index + 1]
            dst_index = batch.dst_index[index : index + 1]
            raw_message = event_features[index : index + 1]

            src_memory = self.memory(src_index).squeeze(0)
            dst_memory = self.memory(dst_index).squeeze(0)
            logits[index] = self.linear(torch.cat([src_memory, dst_memory, raw_message.squeeze(0)], dim=-1))

            self.memory.update(
                src_index=src_index,
                dst_index=dst_index,
                timestamp=batch.timestamp[index : index + 1],
                raw_message=raw_message,
            )
        return logits


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
        TemporalEventRecord(
            graph=graph,
            src_index=0,
            dst_index=1,
            timestamp=3,
            label=1,
            event_features=torch.tensor([1.0, 0.0]),
        ),
        TemporalEventRecord(
            graph=graph,
            src_index=2,
            dst_index=0,
            timestamp=5,
            label=0,
            event_features=torch.tensor([0.0, 1.0]),
        ),
    ]
    return DataLoader(
        dataset=ListDataset(samples),
        sampler=FullGraphSampler(),
        batch_size=2,
    )


def main():
    trainer = Trainer(
        model=TinyTemporalMemoryModel(),
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
