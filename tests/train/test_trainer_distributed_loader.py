from dataclasses import dataclass

import torch
from torch import nn

from vgl import Graph
from vgl.data.dataset import ListDataset
from vgl.data.loader import Loader
from vgl.data.sample import SampleRecord
from vgl.data.sampler import FullGraphSampler
from vgl.distributed import ShardRoute
from vgl.train.tasks import GraphClassificationTask
from vgl.train.trainer import Trainer


@dataclass(slots=True)
class RichBatch:
    base: object
    shard_routes: tuple[ShardRoute, ...]
    partition_ids: torch.Tensor

    def __getattr__(self, name):
        return getattr(self.base, name)


class DistributedAwareLoader:
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        for batch in self.base_loader:
            num_graphs = int(batch.labels.size(0))
            index = torch.arange(num_graphs, dtype=torch.long)
            yield RichBatch(
                base=batch,
                shard_routes=(
                    ShardRoute(
                        partition_id=0,
                        global_ids=index,
                        local_ids=index,
                        positions=index,
                    ),
                ),
                partition_ids=torch.zeros(num_graphs, dtype=torch.long),
            )


class TinyDistributedGraphClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch):
        assert batch.shard_routes[0].partition_id == 0
        return self.linear(torch.randn(batch.labels.size(0), 4))


def _loader():
    samples = [
        SampleRecord(
            graph=Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([1]),
            ),
            metadata={"label": 1},
            sample_id="a",
        ),
        SampleRecord(
            graph=Graph.homo(
                edge_index=torch.tensor([[0], [1]]),
                x=torch.randn(2, 4),
                y=torch.tensor([0]),
            ),
            metadata={"label": 0},
            sample_id="b",
        ),
    ]
    base_loader = Loader(
        dataset=ListDataset(samples),
        sampler=FullGraphSampler(),
        batch_size=2,
        label_source="metadata",
        label_key="label",
    )
    return DistributedAwareLoader(base_loader)


def test_trainer_accepts_distributed_aware_loader_batches_without_api_changes():
    trainer = Trainer(
        model=TinyDistributedGraphClassifier(),
        task=GraphClassificationTask(target="label", label_source="metadata"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    history = trainer.fit(_loader(), val_data=_loader())
    result = trainer.test(_loader())

    assert history["completed_epochs"] == 1
    assert "loss" in result


def test_trainer_moves_distributed_aware_dataclass_batches_to_device():
    trainer = Trainer(
        model=TinyDistributedGraphClassifier(),
        task=GraphClassificationTask(target="label", label_source="metadata"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
        device="cpu",
    )

    history = trainer.fit(_loader(), val_data=_loader())
    result = trainer.test(_loader())

    assert history["completed_epochs"] == 1
    assert "loss" in result
