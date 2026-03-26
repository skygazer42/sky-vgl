import pytest
import torch

from vgl import Graph
from vgl.ops import edge_ids, find_edges, has_edges_between, in_degrees, in_edges, in_subgraph, out_degrees, out_edges, predecessors, reverse, successors


def test_find_edges_returns_endpoints_for_requested_edge_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 1], [1, 0, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    src, dst = find_edges(graph, torch.tensor([2, 0]))

    assert torch.equal(src, torch.tensor([1, 0]))
    assert torch.equal(dst, torch.tensor([2, 1]))


def test_find_edges_uses_public_edge_ids_from_frontier_subgraph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    frontier = in_subgraph(graph, torch.tensor([3, 0]))
    src, dst = find_edges(frontier, torch.tensor([3, 1]))

    assert torch.equal(src, torch.tensor([1, 2]))
    assert torch.equal(dst, torch.tensor([3, 3]))


def test_edge_ids_returns_first_matching_edge_for_each_pair():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 0, 1], [1, 1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    eids = edge_ids(
        graph,
        torch.tensor([0, 0, 1]),
        torch.tensor([1, 2, 2]),
    )

    assert torch.equal(eids, torch.tensor([0, 2, 3]))


def test_edge_ids_return_uv_enumerates_all_matching_edges_in_pair_order():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 0], [1, 1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    src, dst, eids = edge_ids(
        graph,
        torch.tensor([1, 0]),
        torch.tensor([2, 1]),
        return_uv=True,
    )

    assert torch.equal(src, torch.tensor([1, 0, 0]))
    assert torch.equal(dst, torch.tensor([2, 1, 1]))
    assert torch.equal(eids, torch.tensor([2, 0, 1]))


def test_has_edges_between_supports_scalar_and_vector_queries():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    assert has_edges_between(graph, 0, 1) is True
    assert has_edges_between(graph, 1, 0) is False
    assert torch.equal(
        has_edges_between(
            graph,
            torch.tensor([0, 1, 2]),
            torch.tensor([1, 0, 0]),
        ),
        torch.tensor([True, False, True]),
    )


def test_query_ops_reject_missing_pairs_and_invalid_ids():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    with pytest.raises(ValueError):
        find_edges(graph, torch.tensor([3]))

    with pytest.raises(ValueError):
        edge_ids(graph, torch.tensor([2]), torch.tensor([1]))

    with pytest.raises(ValueError):
        has_edges_between(graph, torch.tensor([0, 3]), torch.tensor([1, 1]))


def test_reverse_swaps_homogeneous_edges_and_respects_copy_flags():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"weight": torch.tensor([1.0, 2.0, 3.0])},
    )

    reversed_graph = reverse(graph, copy_ndata=False, copy_edata=False)

    assert torch.equal(reversed_graph.edge_index, torch.tensor([[1, 2, 0], [0, 1, 2]]))
    assert reversed_graph.nodes["node"].data == {}
    assert "weight" not in reversed_graph.edata
    assert torch.equal(reversed_graph.edata["e_id"], torch.tensor([0, 1, 2]))


def test_reverse_preserves_public_edge_ids_on_frontier_subgraph_without_copying_other_edata():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    frontier = in_subgraph(graph, torch.tensor([3, 0]))
    reversed_graph = reverse(frontier, copy_edata=False)

    assert torch.equal(reversed_graph.edge_index, torch.tensor([[3, 0, 3], [2, 3, 1]]))
    assert torch.equal(reversed_graph.edata["e_id"], torch.tensor([1, 2, 3]))
    assert "weight" not in reversed_graph.edata


def test_reverse_flips_heterogeneous_relation_keys_and_copies_features():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    reversed_writes = ("paper", "writes", "author")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[0, 1], [1, 2]]),
                "weight": torch.tensor([0.1, 0.2]),
            },
            cites: {
                "edge_index": torch.tensor([[0, 2], [2, 1]]),
                "score": torch.tensor([4.0, 5.0]),
            },
        },
    )

    reversed_graph = reverse(graph, copy_edata=True)

    assert set(reversed_graph.nodes) == {"author", "paper"}
    assert set(reversed_graph.edges) == {reversed_writes, cites}
    assert torch.equal(reversed_graph.nodes["author"].x, graph.nodes["author"].x)
    assert torch.equal(reversed_graph.nodes["paper"].x, graph.nodes["paper"].x)
    assert torch.equal(
        reversed_graph.edges[reversed_writes].edge_index,
        torch.tensor([[1, 2], [0, 1]]),
    )
    assert torch.equal(reversed_graph.edges[reversed_writes].weight, torch.tensor([0.1, 0.2]))
    assert torch.equal(reversed_graph.edges[reversed_writes].e_id, torch.tensor([0, 1]))
    assert torch.equal(reversed_graph.edges[cites].edge_index, torch.tensor([[2, 1], [0, 2]]))
    assert torch.equal(reversed_graph.edges[cites].score, torch.tensor([4.0, 5.0]))


def test_in_edges_and_out_edges_support_common_forms():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    in_src, in_dst = in_edges(graph, torch.tensor([3, 0]))
    out_eids = out_edges(graph, torch.tensor([0, 3]), form="eid")
    all_src, all_dst, all_eids = in_edges(graph, 3, form="all")

    assert torch.equal(in_src, torch.tensor([2, 3, 1]))
    assert torch.equal(in_dst, torch.tensor([3, 0, 3]))
    assert torch.equal(out_eids, torch.tensor([0, 2]))
    assert torch.equal(all_src, torch.tensor([2, 1]))
    assert torch.equal(all_dst, torch.tensor([3, 3]))
    assert torch.equal(all_eids, torch.tensor([1, 3]))


def test_adjacency_edge_queries_use_public_edge_ids_from_frontier_subgraph():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"weight": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )

    frontier = in_subgraph(graph, torch.tensor([3, 0]))
    src, dst, eids = in_edges(frontier, 3, form="all")
    outbound = out_edges(frontier, torch.tensor([2, 1]), form="eid")

    assert torch.equal(src, torch.tensor([2, 1]))
    assert torch.equal(dst, torch.tensor([3, 3]))
    assert torch.equal(eids, torch.tensor([1, 3]))
    assert torch.equal(outbound, torch.tensor([1, 3]))


def test_predecessors_and_successors_preserve_parallel_edge_duplicates():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    assert torch.equal(predecessors(graph, 1), torch.tensor([0, 0]))
    assert torch.equal(successors(graph, 1), torch.tensor([2, 2]))


def test_selected_relation_adjacency_queries_work_on_heterogeneous_graphs():
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]])},
        },
    )

    src, dst, eids = out_edges(graph, torch.tensor([1]), form="all", edge_type=writes)

    assert torch.equal(src, torch.tensor([1, 1]))
    assert torch.equal(dst, torch.tensor([0, 2]))
    assert torch.equal(eids, torch.tensor([1, 2]))
    assert torch.equal(predecessors(graph, 2, edge_type=writes), torch.tensor([1]))
    assert torch.equal(successors(graph, 1, edge_type=writes), torch.tensor([0, 2]))


def test_adjacency_query_ops_validate_form_and_node_ranges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    with pytest.raises(ValueError):
        in_edges(graph, torch.tensor([1]), form="bad")

    with pytest.raises(ValueError):
        out_edges(graph, torch.tensor([3]), form="eid")

    with pytest.raises(ValueError):
        predecessors(graph, 3)

    with pytest.raises(ValueError):
        successors(graph, 3)


def test_in_degrees_and_out_degrees_support_scalar_vector_and_all_node_queries():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    assert in_degrees(graph, 2) == 2
    assert out_degrees(graph, 0) == 2
    assert torch.equal(in_degrees(graph), torch.tensor([1, 1, 2, 0]))
    assert torch.equal(out_degrees(graph), torch.tensor([2, 1, 1, 0]))
    assert torch.equal(in_degrees(graph, torch.tensor([3, 2, 3])), torch.tensor([0, 2, 0]))
    assert torch.equal(out_degrees(graph, torch.tensor([0, 3, 0])), torch.tensor([2, 0, 2]))


def test_in_degrees_and_out_degrees_work_on_selected_heterogeneous_relation():
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]])},
        },
    )

    assert in_degrees(graph, 2, edge_type=writes) == 1
    assert out_degrees(graph, 1, edge_type=writes) == 2
    assert torch.equal(in_degrees(graph, edge_type=writes), torch.tensor([1, 1, 1]))
    assert torch.equal(out_degrees(graph, edge_type=writes), torch.tensor([1, 2]))


def test_in_degrees_and_out_degrees_validate_node_ranges():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    with pytest.raises(ValueError):
        in_degrees(graph, torch.tensor([3]))

    with pytest.raises(ValueError):
        out_degrees(graph, torch.tensor([3]))
