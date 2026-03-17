import torch

from examples.homo.link_prediction import TinyLinkPredictor, build_demo_loader
from vgl.engine import Trainer
from vgl.tasks import LinkPredictionTask


def test_end_to_end_link_prediction_runs():
    trainer = Trainer(
        model=TinyLinkPredictor(),
        task=LinkPredictionTask(target="label"),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=1,
    )

    result = trainer.fit(build_demo_loader())

    assert result["epochs"] == 1
