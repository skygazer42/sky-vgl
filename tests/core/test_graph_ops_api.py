import torch

from vgl import Graph


def test_graph_method_bridges_call_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    with_loops = graph.add_self_loops()
    subgraph = graph.node_subgraph(torch.tensor([0, 1, 2]))
    compacted, mapping = graph.compact_nodes(torch.tensor([0, 1, 2]))
    line = graph.line_graph(copy_edata=False)

    assert {tuple(edge) for edge in with_loops.edge_index.t().tolist()} == {(0, 1), (1, 2), (0, 0), (1, 1), (2, 2)}
    assert torch.equal(subgraph.edge_index, graph.edge_index)
    assert torch.equal(compacted.edge_index, graph.edge_index)
    assert mapping == {0: 0, 1: 1, 2: 2}
    assert torch.equal(line.n_id, torch.tensor([0, 1]))
    assert torch.equal(line.edge_index, torch.tensor([[0], [1]]))


def test_graph_metapath_reachable_graph_bridge_calls_ops_layer():
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [0, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    metapath = graph.metapath_reachable_graph([writes, published_in])
    edge_type = ("author", "writes__published_in", "venue")

    assert set(metapath.nodes) == {"author", "venue"}
    assert set(metapath.edges) == {edge_type}
    assert torch.equal(metapath.edges[edge_type].edge_index, torch.tensor([[0, 1], [0, 0]]))


def test_graph_random_walk_bridge_calls_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    assert hasattr(graph, "random_walk")
    traces = graph.random_walk(torch.tensor([0, 2]), length=3)

    assert torch.equal(traces, torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]]))


def test_graph_to_block_bridge_calls_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    block = graph.to_block(torch.tensor([1, 2]))

    assert torch.equal(block.dst_n_id, torch.tensor([1, 2]))
    assert torch.equal(block.src_n_id, torch.tensor([1, 2, 0]))
    assert torch.equal(block.edge_index, torch.tensor([[2, 0, 1], [0, 1, 0]]))


def test_graph_metapath_random_walk_bridge_calls_ops_layer():
    writes = ("author", "writes", "paper")
    published_in = ("paper", "published_in", "venue")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0]])},
            "venue": {"x": torch.tensor([[100.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [0, 1]])},
            published_in: {"edge_index": torch.tensor([[0, 1], [0, 0]])},
        },
    )

    assert hasattr(graph, "metapath_random_walk")
    traces = graph.metapath_random_walk(torch.tensor([0, 1]), [writes, published_in])

    assert torch.equal(traces, torch.tensor([[0, 0, 0], [1, 1, 0]]))
