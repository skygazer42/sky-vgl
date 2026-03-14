import torch

from examples.homo.node_classification import TinyHomoModel, build_demo_graph
from vgl import NodeClassificationTask, Trainer


def test_end_to_end_homo_training_loop_runs(tmp_path):
    graph = build_demo_graph()
    trainer = Trainer(
        model=TinyHomoModel(),
        task=NodeClassificationTask(
            target="y",
            split=("train_mask", "val_mask", "test_mask"),
            metrics=["accuracy"],
        ),
        optimizer=torch.optim.Adam,
        lr=1e-2,
        max_epochs=2,
        monitor="val_accuracy",
        save_best_path=tmp_path / "best.pt",
    )

    history = trainer.fit(graph, val_data=graph)
    test_result = trainer.test(graph)

    assert history["epochs"] == 2
    assert len(history["train"]) == 2
    assert len(history["val"]) == 2
    assert "accuracy" in history["val"][-1]
    assert history["best_epoch"] is not None
    assert "accuracy" in test_result

