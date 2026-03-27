import torch

from vgl import Graph
from vgl.sparse import SparseLayout, to_coo


def _sparse_to_dense(sparse) -> torch.Tensor:
    coo = to_coo(sparse)
    dtype = torch.float32 if coo.values is None else coo.values.dtype
    dense = torch.zeros(coo.shape, dtype=dtype)
    if coo.nnz == 0:
        return dense
    values = torch.ones(coo.nnz, dtype=dtype) if coo.values is None else coo.values.cpu()
    dense[coo.row.cpu(), coo.col.cpu()] = values
    return dense


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


def test_graph_to_hetero_block_bridge_calls_ops_layer():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[10.0], [20.0]])},
            "paper": {"x": torch.tensor([[1.0], [2.0], [3.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]])},
            cites: {"edge_index": torch.tensor([[0, 1, 2], [2, 2, 0]])},
        },
    )

    block = graph.to_hetero_block({"paper": torch.tensor([0, 2])})

    assert torch.equal(block.dst_n_id["paper"], torch.tensor([0, 2]))
    assert torch.equal(block.src_n_id["author"], torch.tensor([1]))
    assert torch.equal(block.src_n_id["paper"], torch.tensor([0, 2, 1]))
    assert torch.equal(block.edge_index(writes), torch.tensor([[0, 0], [0, 1]]))
    assert torch.equal(block.edge_index(cites), torch.tensor([[0, 2, 1], [1, 1, 0]]))


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


def test_graph_frontier_subgraph_bridges_call_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"edge_weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    inbound = graph.in_subgraph(torch.tensor([3, 0]))
    outbound = graph.out_subgraph(torch.tensor([0, 3]))

    assert torch.equal(inbound.edge_index, torch.tensor([[2, 3, 1], [3, 0, 3]]))
    assert torch.equal(inbound.edata["e_id"], torch.tensor([1, 2, 3]))
    assert torch.equal(outbound.edge_index, torch.tensor([[0, 3], [2, 0]]))
    assert torch.equal(outbound.edata["e_id"], torch.tensor([0, 2]))


def test_graph_query_and_reverse_bridges_call_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    src, dst = graph.find_edges(torch.tensor([3, 0]))
    eids = graph.edge_ids(torch.tensor([0, 1]), torch.tensor([1, 2]))
    connected = graph.has_edges_between(torch.tensor([0, 2]), torch.tensor([1, 1]))
    reversed_graph = graph.reverse(copy_edata=False)

    assert torch.equal(src, torch.tensor([2, 0]))
    assert torch.equal(dst, torch.tensor([0, 1]))
    assert torch.equal(eids, torch.tensor([0, 2]))
    assert torch.equal(connected, torch.tensor([True, False]))
    assert torch.equal(reversed_graph.edge_index, torch.tensor([[1, 1, 2, 0], [0, 0, 1, 2]]))
    assert torch.equal(reversed_graph.edata["e_id"], torch.tensor([0, 1, 2, 3]))


def test_graph_to_simple_bridge_calls_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 0], [1, 1, 2, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    simplified = graph.to_simple(count_attr="count")

    assert torch.equal(simplified.edge_index, torch.tensor([[0, 1], [1, 2]]))
    assert torch.equal(simplified.edata["weight"], torch.tensor([1.0, 3.0]))
    assert torch.equal(simplified.edata["count"], torch.tensor([3, 1]))


def test_graph_adjacency_query_bridges_call_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    in_src, in_dst = graph.in_edges(torch.tensor([1]))
    out_eids = graph.out_edges(torch.tensor([1]), form="eid")
    preds = graph.predecessors(1)
    succs = graph.successors(1)

    assert torch.equal(in_src, torch.tensor([0, 0]))
    assert torch.equal(in_dst, torch.tensor([1, 1]))
    assert torch.equal(out_eids, torch.tensor([2, 3]))
    assert torch.equal(preds, torch.tensor([0, 0]))
    assert torch.equal(succs, torch.tensor([2, 2]))


def test_graph_laplacian_bridge_calls_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    result = graph.laplacian(normalization="rw")

    assert result.layout is SparseLayout.COO
    assert torch.allclose(
        _sparse_to_dense(result),
        torch.tensor(
            [
                [1.0, -1.0, 0.0],
                [-0.5, 1.0, -0.5],
                [0.0, -1.0, 1.0],
            ]
        ),
    )


def test_graph_in_degrees_and_out_degrees_bridges_call_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    assert graph.in_degrees(2) == 2
    assert graph.out_degrees(0) == 2
    assert torch.equal(graph.in_degrees(), torch.tensor([1, 1, 2, 0]))
    assert torch.equal(graph.out_degrees(torch.tensor([0, 3])), torch.tensor([2, 0]))


def test_graph_cardinality_and_all_edges_bridges_call_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [0, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"e_id": torch.tensor([0, 2, 1])},
    )

    assert graph.num_nodes() == 3
    assert graph.number_of_nodes() == 3
    assert graph.num_edges() == 3
    assert graph.number_of_edges() == 3
    src, dst, eids = graph.all_edges(form="all", order="srcdst")

    assert torch.equal(src, torch.tensor([0, 1, 2]))
    assert torch.equal(dst, torch.tensor([1, 0, 0]))
    assert torch.equal(eids, torch.tensor([2, 1, 0]))


def test_graph_inc_bridge_calls_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 2, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    incidence = graph.inc("both", layout=SparseLayout.CSC)

    assert incidence.layout is SparseLayout.CSC
    assert incidence.shape == (3, 3)
    assert torch.equal(
        _sparse_to_dense(incidence),
        torch.tensor(
            [
                [-1.0, 0.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    )


def test_graph_adj_tensors_bridge_calls_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    crow_indices, col_indices, eids = graph.adj_tensors("csr")

    assert torch.equal(crow_indices, torch.tensor([0, 1, 2, 3, 3]))
    assert torch.equal(col_indices, torch.tensor([1, 0, 1]))
    assert torch.equal(eids, torch.tensor([1, 2, 0]))


def test_graph_adj_bridge_calls_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"weight": torch.tensor([0.5, 1.5, 2.5])},
    )

    weighted = graph.adj(eweight_name="weight", layout=SparseLayout.CSC)

    assert weighted.layout is SparseLayout.CSC
    assert weighted.shape == (4, 4)
    assert torch.equal(
        _sparse_to_dense(weighted),
        torch.tensor(
            [
                [0.0, 1.5, 0.0, 0.0],
                [2.5, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_graph_adj_external_bridge_calls_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    adjacency = graph.adj_external(transpose=True)

    assert adjacency.layout is torch.sparse_coo
    assert tuple(adjacency.size()) == (4, 4)
    assert torch.equal(adjacency._indices(), torch.tensor([[1, 1, 0], [2, 0, 1]]))


def test_graph_adj_external_bridge_supports_torch_format_keyword():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    adjacency = graph.adj_external(torch_fmt="csr")

    assert adjacency.layout is torch.sparse_csr
    assert tuple(adjacency.size()) == (4, 4)
    assert torch.equal(adjacency.crow_indices(), torch.tensor([0, 1, 2, 3, 3]))
    assert torch.equal(adjacency.col_indices(), torch.tensor([1, 0, 1]))


def test_graph_formats_and_create_formats_bridge_call_ops_layer():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1], [1, 0, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    clone = graph.formats(["coo", "csr"])
    result = clone.create_formats_()

    assert result is None
    assert graph.formats() == {"created": ["coo"], "not created": ["csr", "csc"]}
    assert clone.formats() == {"created": ["coo", "csr"], "not created": []}
