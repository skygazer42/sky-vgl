from vgl import (
    Accuracy,
    APPNPConv,
    ChebConv,
    GATv2Conv,
    Graph,
    GraphBatch,
    GraphSchema,
    GraphView,
    GINConv,
    LinkPredictionBatch,
    LinkPredictionRecord,
    LinkPredictionTask,
    ListDataset,
    Loader,
    FullGraphSampler,
    NodeSeedSubgraphSampler,
    SampleRecord,
    TemporalEventRecord,
    MessagePassing,
    Task,
    Metric,
    SGConv,
    TAGConv,
    Trainer,
    NodeClassificationTask,
    GraphClassificationTask,
    TemporalEventPredictionTask,
    TemporalEventBatch,
    global_mean_pool,
    global_sum_pool,
    global_max_pool,
    __version__,
)


def test_package_exposes_broad_vgl_root_surface():
    assert Accuracy.__name__ == "Accuracy"
    assert APPNPConv.__name__ == "APPNPConv"
    assert ChebConv.__name__ == "ChebConv"
    assert GATv2Conv.__name__ == "GATv2Conv"
    assert Graph.__name__ == "Graph"
    assert GraphBatch.__name__ == "GraphBatch"
    assert LinkPredictionBatch.__name__ == "LinkPredictionBatch"
    assert TemporalEventBatch.__name__ == "TemporalEventBatch"
    assert GINConv.__name__ == "GINConv"
    assert GraphSchema.__name__ == "GraphSchema"
    assert GraphView.__name__ == "GraphView"
    assert ListDataset.__name__ == "ListDataset"
    assert Loader.__name__ == "Loader"
    assert FullGraphSampler.__name__ == "FullGraphSampler"
    assert NodeSeedSubgraphSampler.__name__ == "NodeSeedSubgraphSampler"
    assert LinkPredictionRecord.__name__ == "LinkPredictionRecord"
    assert SampleRecord.__name__ == "SampleRecord"
    assert TemporalEventRecord.__name__ == "TemporalEventRecord"
    assert MessagePassing.__name__ == "MessagePassing"
    assert SGConv.__name__ == "SGConv"
    assert TAGConv.__name__ == "TAGConv"
    assert Task.__name__ == "Task"
    assert Metric.__name__ == "Metric"
    assert Trainer.__name__ == "Trainer"
    assert NodeClassificationTask.__name__ == "NodeClassificationTask"
    assert GraphClassificationTask.__name__ == "GraphClassificationTask"
    assert LinkPredictionTask.__name__ == "LinkPredictionTask"
    assert TemporalEventPredictionTask.__name__ == "TemporalEventPredictionTask"
    assert callable(global_mean_pool)
    assert callable(global_sum_pool)
    assert callable(global_max_pool)
    assert __version__ == "0.1.0"
