import torch

from examples.temporal.event_prediction import TinyTemporalEventModel, build_demo_loader
from vgl.engine import Trainer
from vgl.tasks import TemporalEventPredictionTask


def test_end_to_end_temporal_event_prediction_runs():
    loader = build_demo_loader()
    trainer = Trainer(
        model=TinyTemporalEventModel(),
        task=TemporalEventPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(loader)

    assert result["epochs"] == 1
