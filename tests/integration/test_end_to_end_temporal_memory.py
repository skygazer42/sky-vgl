import torch

from examples.temporal.memory_event_prediction import TinyTemporalMemoryModel, build_demo_loader
from vgl.engine import Trainer
from vgl.tasks import TemporalEventPredictionTask


def test_end_to_end_temporal_memory_event_prediction_runs():
    loader = build_demo_loader()
    trainer = Trainer(
        model=TinyTemporalMemoryModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(loader)

    assert result["epochs"] == 1
