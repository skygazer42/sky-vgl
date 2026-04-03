import pytest
import scipy.sparse
import torch

from vgl import Graph
from vgl.ops import adj, adj_external, adj_tensors, all_edges, edge_ids, find_edges, has_edges_between, in_degrees, in_edges, in_subgraph, laplacian, num_edges, num_nodes, number_of_edges, number_of_nodes, out_degrees, out_edges, out_subgraph, predecessors, reverse, successors
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


def test_query_ops_populate_and_reuse_edge_lookup_cache():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 0], [1, 1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )
    store = graph.edges[("node", "to", "node")]

    assert getattr(store, "query_cache") == {}

    src, dst = find_edges(graph, torch.tensor([2, 0]))
    assert torch.equal(src, torch.tensor([1, 0]))
    assert torch.equal(dst, torch.tensor([2, 1]))
    assert store.query_cache

    cache_entries = dict(store.query_cache)

    eids = edge_ids(graph, torch.tensor([0, 1]), torch.tensor([1, 2]))
    assert torch.equal(eids, torch.tensor([0, 2]))
    assert set(store.query_cache) >= set(cache_entries)
    for key, value in cache_entries.items():
        assert store.query_cache[key] is value

    assert torch.equal(
        has_edges_between(graph, torch.tensor([0, 1]), torch.tensor([2, 2])),
        torch.tensor([True, True]),
    )
    for key, value in store.query_cache.items():
        assert store.query_cache[key] is value


def test_find_edges_avoids_right_boundary_searchsorted(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 1], [1, 0, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )
    original_searchsorted = torch.searchsorted
    right_flags = []

    def record_searchsorted(sorted_sequence, input, *, out_int32=False, right=False, side=None, out=None, sorter=None):
        right_flags.append(right if side is None else side == "right")
        return original_searchsorted(
            sorted_sequence,
            input,
            out_int32=out_int32,
            right=right,
            side=side,
            out=out,
            sorter=sorter,
        )

    monkeypatch.setattr(torch, "searchsorted", record_searchsorted)

    src, dst = find_edges(graph, torch.tensor([2, 0]))

    assert torch.equal(src, torch.tensor([1, 0]))
    assert torch.equal(dst, torch.tensor([2, 1]))
    assert right_flags
    assert not any(right_flags)


def test_edge_ids_without_return_uv_avoids_right_boundary_searchsorted(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 0, 1], [1, 1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )
    original_searchsorted = torch.searchsorted
    right_flags = []

    def record_searchsorted(sorted_sequence, input, *, out_int32=False, right=False, side=None, out=None, sorter=None):
        right_flags.append(right if side is None else side == "right")
        return original_searchsorted(
            sorted_sequence,
            input,
            out_int32=out_int32,
            right=right,
            side=side,
            out=out,
            sorter=sorter,
        )

    monkeypatch.setattr(torch, "searchsorted", record_searchsorted)

    eids = edge_ids(
        graph,
        torch.tensor([0, 0, 1]),
        torch.tensor([1, 2, 2]),
    )

    assert torch.equal(eids, torch.tensor([0, 2, 3]))
    assert right_flags
    assert not any(right_flags)


def test_has_edges_between_avoids_right_boundary_searchsorted(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )
    original_searchsorted = torch.searchsorted
    right_flags = []

    def record_searchsorted(sorted_sequence, input, *, out_int32=False, right=False, side=None, out=None, sorter=None):
        right_flags.append(right if side is None else side == "right")
        return original_searchsorted(
            sorted_sequence,
            input,
            out_int32=out_int32,
            right=right,
            side=side,
            out=out,
            sorter=sorter,
        )

    monkeypatch.setattr(torch, "searchsorted", record_searchsorted)

    exists = has_edges_between(
        graph,
        torch.tensor([0, 1, 2]),
        torch.tensor([1, 0, 0]),
    )

    assert torch.equal(exists, torch.tensor([True, False, True]))
    assert right_flags
    assert not any(right_flags)


def test_query_match_checks_avoid_dense_zero_buffers(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )
    original_zeros = torch.zeros

    def fail_dense_bool_zeros(*args, **kwargs):
        dtype = kwargs.get("dtype")
        if dtype is torch.bool:
            raise AssertionError("query matches should avoid allocating dense bool zero buffers")
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", fail_dense_bool_zeros)

    src, dst = find_edges(graph, torch.tensor([3, 0]))
    eids = edge_ids(graph, torch.tensor([0, 1]), torch.tensor([2, 2]))
    exists = has_edges_between(graph, torch.tensor([0, 2]), torch.tensor([1, 0]))

    assert torch.equal(src, torch.tensor([2, 0]))
    assert torch.equal(dst, torch.tensor([0, 1]))
    assert torch.equal(eids, torch.tensor([1, 2]))
    assert torch.equal(exists, torch.tensor([True, True]))


def test_edge_ids_return_uv_avoids_repeat_interleave(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 0], [1, 1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("edge_ids(return_uv=True) should avoid repeat_interleave interval expansion")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    src, dst, eids = edge_ids(
        graph,
        torch.tensor([1, 0]),
        torch.tensor([2, 1]),
        return_uv=True,
    )

    assert torch.equal(src, torch.tensor([1, 0, 0]))
    assert torch.equal(dst, torch.tensor([2, 1, 1]))
    assert torch.equal(eids, torch.tensor([2, 0, 1]))


def test_endpoint_edge_queries_avoid_repeat_interleave(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    def fail_repeat_interleave(*args, **kwargs):
        raise AssertionError("endpoint edge queries should avoid repeat_interleave interval expansion")

    monkeypatch.setattr(torch, "repeat_interleave", fail_repeat_interleave)

    in_src, in_dst = in_edges(graph, torch.tensor([3, 0]))
    out_eids = out_edges(graph, torch.tensor([0, 3]), form="eid")

    assert torch.equal(in_src, torch.tensor([2, 3, 1]))
    assert torch.equal(in_dst, torch.tensor([3, 0, 3]))
    assert torch.equal(out_eids, torch.tensor([0, 2]))


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


def test_in_edges_and_out_edges_avoid_torch_isin(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 3, 1], [2, 3, 0, 3]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("endpoint edge queries should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    in_src, in_dst = in_edges(graph, torch.tensor([3, 0]))
    out_eids = out_edges(graph, torch.tensor([0, 3]), form="eid")

    assert torch.equal(in_src, torch.tensor([2, 3, 1]))
    assert torch.equal(in_dst, torch.tensor([3, 0, 3]))
    assert torch.equal(out_eids, torch.tensor([0, 2]))


def test_predecessors_and_successors_avoid_torch_isin(monkeypatch):
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 1, 1], [1, 1, 2, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    def fail_isin(*args, **kwargs):
        raise AssertionError("predecessor/successor queries should avoid torch.isin scans")

    monkeypatch.setattr(torch, "isin", fail_isin)

    assert torch.equal(predecessors(graph, 1), torch.tensor([0, 0]))
    assert torch.equal(successors(graph, 1), torch.tensor([2, 2]))


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


def test_num_nodes_and_num_edges_support_total_and_type_specific_counts():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            cites: {"edge_index": torch.tensor([[0], [2]])},
        },
    )

    assert num_nodes(graph) == 5
    assert number_of_nodes(graph, "author") == 2
    assert number_of_nodes(graph, "paper") == 3
    assert num_edges(graph) == 3
    assert number_of_edges(graph, writes) == 2
    assert number_of_edges(graph, cites) == 1


def test_all_edges_supports_forms_and_public_eid_ordering():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [0, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"e_id": torch.tensor([0, 2, 1])},
    )

    src, dst = all_edges(graph)
    eids = all_edges(graph, form="eid")
    src_all, dst_all, all_eids = all_edges(graph, form="all", order="srcdst")

    assert torch.equal(src, torch.tensor([2, 1, 0]))
    assert torch.equal(dst, torch.tensor([0, 0, 1]))
    assert torch.equal(eids, torch.tensor([0, 1, 2]))
    assert torch.equal(src_all, torch.tensor([0, 1, 2]))
    assert torch.equal(dst_all, torch.tensor([1, 0, 0]))
    assert torch.equal(all_eids, torch.tensor([2, 1, 0]))


def test_all_edges_requires_edge_type_for_multi_relation_graphs():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            cites: {"edge_index": torch.tensor([[0], [2]])},
        },
    )

    with pytest.raises(ValueError):
        all_edges(graph)

    src, dst, eids = all_edges(graph, form="all", edge_type=writes)

    assert torch.equal(src, torch.tensor([0, 1]))
    assert torch.equal(dst, torch.tensor([1, 2]))
    assert torch.equal(eids, torch.tensor([0, 1]))


def test_all_edges_validates_form_and_order():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    with pytest.raises(ValueError):
        all_edges(graph, form="bad")

    with pytest.raises(ValueError):
        all_edges(graph, order="bad")


def test_incidence_sparse_view_supports_in_out_and_both_types():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    inbound = graph.inc("in")
    outbound = graph.inc("out")
    both = graph.inc("both", layout=SparseLayout.CSR)

    assert inbound.layout is SparseLayout.COO
    assert inbound.shape == (3, 2)
    assert torch.equal(_sparse_to_dense(inbound), torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
    assert torch.equal(_sparse_to_dense(outbound), torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
    assert both.layout is SparseLayout.CSR
    assert torch.equal(_sparse_to_dense(both), torch.tensor([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]]))


def test_incidence_sparse_view_uses_public_edge_id_column_order_and_ignores_self_loops_for_both():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1, 1], [0, 1, 2, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"e_id": torch.tensor([0, 2, 1, 3])},
    )

    outbound = graph.inc("out")
    both = graph.inc("both")

    assert torch.equal(
        _sparse_to_dense(outbound),
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(
        _sparse_to_dense(both),
        torch.tensor(
            [
                [1.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0, 0.0],
            ]
        ),
    )


def test_incidence_sparse_view_rejects_both_for_bipartite_relation_and_invalid_typestr():
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
        },
    )

    with pytest.raises(ValueError):
        graph.inc("both", edge_type=writes)

    with pytest.raises(ValueError):
        graph.inc("bad")


def test_adj_tensors_support_coo_csr_and_csc_layouts():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    src, dst = adj_tensors(graph, "coo")
    crow_indices, col_indices, csr_eids = adj_tensors(graph, "csr")
    ccol_indices, row_indices, csc_eids = adj_tensors(graph, "csc")

    assert torch.equal(src, torch.tensor([2, 0, 1]))
    assert torch.equal(dst, torch.tensor([1, 1, 0]))
    assert torch.equal(crow_indices, torch.tensor([0, 1, 2, 3, 3]))
    assert torch.equal(col_indices, torch.tensor([1, 0, 1]))
    assert torch.equal(csr_eids, torch.tensor([1, 2, 0]))
    assert torch.equal(ccol_indices, torch.tensor([0, 1, 3, 3, 3]))
    assert torch.equal(row_indices, torch.tensor([1, 2, 0]))
    assert torch.equal(csc_eids, torch.tensor([2, 0, 1]))


def test_adj_tensors_use_public_edge_ids_for_derived_graph_views():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 1, 0], [2, 3, 0, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"e_id": torch.tensor([5, 1, 7, 3])},
    )

    frontier = out_subgraph(graph, torch.tensor([0, 2]))
    src, dst = adj_tensors(frontier, "coo")
    crow_indices, col_indices, eids = adj_tensors(frontier, "csr")

    assert torch.equal(src, torch.tensor([2, 0, 0]))
    assert torch.equal(dst, torch.tensor([3, 1, 2]))
    assert torch.equal(crow_indices, torch.tensor([0, 2, 2, 3, 3]))
    assert torch.equal(col_indices, torch.tensor([1, 2, 3]))
    assert torch.equal(eids, torch.tensor([3, 5, 1]))


def test_adj_tensors_preserve_public_edge_order_within_compressed_buckets():
    csr_graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 0], [2, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )
    csc_graph = Graph.homo(
        edge_index=torch.tensor([[2, 1, 0], [0, 0, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    crow_indices, col_indices, csr_eids = adj_tensors(csr_graph, "csr")
    ccol_indices, row_indices, csc_eids = adj_tensors(csc_graph, "csc")

    assert torch.equal(crow_indices, torch.tensor([0, 3, 3, 3]))
    assert torch.equal(col_indices, torch.tensor([2, 1, 0]))
    assert torch.equal(csr_eids, torch.tensor([0, 1, 2]))
    assert torch.equal(ccol_indices, torch.tensor([0, 3, 3, 3]))
    assert torch.equal(row_indices, torch.tensor([2, 1, 0]))
    assert torch.equal(csc_eids, torch.tensor([0, 1, 2]))


def test_adj_tensors_support_selected_heterogeneous_relation_and_validate_layout():
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[1, 0], [2, 1]])},
        },
    )

    src, dst = adj_tensors(graph, edge_type=writes)

    assert torch.equal(src, torch.tensor([1, 0]))
    assert torch.equal(dst, torch.tensor([2, 1]))

    with pytest.raises(ValueError):
        adj_tensors(graph, "bad", edge_type=writes)


def test_adj_returns_sparse_tensor_with_optional_edge_weights():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={"weight": torch.tensor([0.5, 1.5, 2.5])},
    )

    adjacency = adj(graph)
    weighted = adj(graph, eweight_name="weight")

    assert adjacency.layout is SparseLayout.COO
    assert adjacency.shape == (4, 4)
    assert torch.equal(
        _sparse_to_dense(adjacency),
        torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
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


def test_adj_uses_public_edge_ids_for_visible_coo_order_and_weight_alignment():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 2, 1, 0], [2, 3, 0, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
        edge_data={
            "e_id": torch.tensor([5, 1, 7, 3]),
            "weight": torch.tensor([5.0, 1.0, 7.0, 3.0]),
        },
    )

    frontier = out_subgraph(graph, torch.tensor([0, 2]))
    weighted = adj(frontier, eweight_name="weight")
    coo = to_coo(weighted)

    assert torch.equal(coo.row, torch.tensor([2, 0, 0]))
    assert torch.equal(coo.col, torch.tensor([3, 1, 2]))
    assert torch.equal(coo.values, torch.tensor([1.0, 3.0, 5.0]))


def test_adj_preserves_public_edge_order_within_csr_and_csc_buckets():
    csr_graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 0], [2, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"weight": torch.tensor([2.0, 1.0, 0.5])},
    )
    csc_graph = Graph.homo(
        edge_index=torch.tensor([[2, 1, 0], [0, 0, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"weight": torch.tensor([2.0, 1.0, 0.5])},
    )

    csr = adj(csr_graph, eweight_name="weight", layout="csr")
    csc = adj(csc_graph, eweight_name="weight", layout="csc")

    assert csr.layout is SparseLayout.CSR
    assert torch.equal(csr.crow_indices, torch.tensor([0, 3, 3, 3]))
    assert torch.equal(csr.col_indices, torch.tensor([2, 1, 0]))
    assert torch.equal(csr.values, torch.tensor([2.0, 1.0, 0.5]))
    assert csc.layout is SparseLayout.CSC
    assert torch.equal(csc.ccol_indices, torch.tensor([0, 3, 3, 3]))
    assert torch.equal(csc.row_indices, torch.tensor([2, 1, 0]))
    assert torch.equal(csc.values, torch.tensor([2.0, 1.0, 0.5]))


def test_adj_supports_selected_heterogeneous_relation_and_validates_weight_name():
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {
                "edge_index": torch.tensor([[1, 0], [2, 1]]),
                "weight": torch.tensor([4.0, 5.0]),
            },
        },
    )

    weighted = adj(graph, edge_type=writes, eweight_name="weight")

    assert weighted.shape == (2, 3)
    assert torch.equal(
        _sparse_to_dense(weighted),
        torch.tensor(
            [
                [0.0, 5.0, 0.0],
                [0.0, 0.0, 4.0],
            ]
        ),
    )

    with pytest.raises(ValueError):
        adj(graph, edge_type=writes, eweight_name="missing")


def test_laplacian_returns_weighted_degree_minus_adjacency_and_aggregates_duplicates():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 0, 0, 1, 1], [1, 1, 0, 2, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
        edge_data={"weight": torch.tensor([1.0, 2.0, 4.0, 3.0, 5.0])},
    )

    result = laplacian(graph, eweight_name="weight")

    assert result.layout is SparseLayout.COO
    assert result.shape == (3, 3)
    assert torch.equal(
        _sparse_to_dense(result),
        torch.tensor(
            [
                [3.0, -3.0, 0.0],
                [0.0, 3.0, -3.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )


def test_laplacian_supports_random_walk_and_symmetric_normalization():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    random_walk = laplacian(graph, normalization="rw")
    symmetric = laplacian(graph, normalization="sym")

    expected = 1.0 / torch.sqrt(torch.tensor(2.0))
    assert torch.allclose(
        _sparse_to_dense(random_walk),
        torch.tensor(
            [
                [1.0, -1.0, 0.0],
                [-0.5, 1.0, -0.5],
                [0.0, -1.0, 1.0],
            ]
        ),
    )
    assert torch.allclose(
        _sparse_to_dense(symmetric),
        torch.tensor(
            [
                [1.0, -expected, 0.0],
                [-expected, 1.0, -expected],
                [0.0, -expected, 1.0],
            ]
        ),
    )


def test_laplacian_supports_square_heterogeneous_relations_and_validates_inputs():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[0, 1], [1, 2]])},
            cites: {
                "edge_index": torch.tensor([[0, 1, 1], [1, 0, 2]]),
                "weight": torch.tensor([2.0, 4.0, 1.0]),
            },
        },
    )

    result = laplacian(graph, edge_type=cites, eweight_name="weight", layout="csr")

    assert result.layout is SparseLayout.CSR
    assert result.shape == (3, 3)
    assert torch.equal(
        _sparse_to_dense(result),
        torch.tensor(
            [
                [2.0, -2.0, 0.0],
                [-4.0, 5.0, -1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
    )

    with pytest.raises(ValueError):
        laplacian(graph, edge_type=writes)

    with pytest.raises(ValueError):
        laplacian(graph, edge_type=cites, normalization="bad")

    with pytest.raises(ValueError):
        laplacian(graph, edge_type=cites, eweight_name="missing")


def test_adj_external_returns_torch_sparse_tensor_and_supports_transpose():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    adjacency = adj_external(graph)
    transposed = adj_external(graph, transpose=True)

    assert adjacency.layout is torch.sparse_coo
    assert tuple(adjacency.size()) == (4, 4)
    assert torch.equal(adjacency._indices(), torch.tensor([[2, 0, 1], [1, 1, 0]]))
    assert torch.equal(adjacency._values(), torch.tensor([1.0, 1.0, 1.0]))
    assert torch.equal(transposed._indices(), torch.tensor([[1, 1, 0], [2, 0, 1]]))
    assert torch.equal(transposed._values(), torch.tensor([1.0, 1.0, 1.0]))


def test_adj_external_supports_scipy_coo_and_csr_exports():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    coo = adj_external(graph, scipy_fmt="coo")
    csr = adj_external(graph, scipy_fmt="csr", transpose=True)

    assert isinstance(coo, scipy.sparse.coo_matrix)
    assert coo.shape == (4, 4)
    assert list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist())) == [(2, 1, 1.0), (0, 1, 1.0), (1, 0, 1.0)]
    assert isinstance(csr, scipy.sparse.csr_matrix)
    assert csr.shape == (4, 4)
    assert list(zip(csr.tocoo().row.tolist(), csr.tocoo().col.tolist(), csr.tocoo().data.tolist())) == [(0, 1, 1.0), (1, 2, 1.0), (1, 0, 1.0)]


def test_adj_external_supports_torch_csr_and_csc_exports():
    graph = Graph.homo(
        edge_index=torch.tensor([[2, 0, 1], [1, 1, 0]]),
        x=torch.tensor([[1.0], [2.0], [3.0], [4.0]]),
    )

    csr = adj_external(graph, torch_fmt="csr")
    csc = adj_external(graph, torch_fmt="csc", transpose=True)

    assert csr.layout is torch.sparse_csr
    assert tuple(csr.size()) == (4, 4)
    assert torch.equal(csr.crow_indices(), torch.tensor([0, 1, 2, 3, 3]))
    assert torch.equal(csr.col_indices(), torch.tensor([1, 0, 1]))
    assert torch.equal(csr.values(), torch.tensor([1.0, 1.0, 1.0]))

    assert csc.layout is torch.sparse_csc
    assert tuple(csc.size()) == (4, 4)
    assert torch.equal(csc.ccol_indices(), torch.tensor([0, 1, 2, 3, 3]))
    assert torch.equal(csc.row_indices(), torch.tensor([1, 0, 1]))
    assert torch.equal(csc.values(), torch.tensor([1.0, 1.0, 1.0]))


def test_adj_external_supports_selected_heterogeneous_relation_and_validates_format():
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[1, 0], [2, 1]])},
        },
    )

    coo = adj_external(graph, edge_type=writes, scipy_fmt="coo")
    transposed = adj_external(graph, edge_type=writes, transpose=True)

    assert coo.shape == (2, 3)
    assert list(zip(coo.row.tolist(), coo.col.tolist(), coo.data.tolist())) == [(1, 2, 1.0), (0, 1, 1.0)]
    assert tuple(transposed.size()) == (3, 2)
    assert torch.equal(transposed._indices(), torch.tensor([[2, 1], [1, 0]]))

    with pytest.raises(ValueError):
        adj_external(graph, edge_type=writes, scipy_fmt="csc")


def test_adj_external_supports_selected_heterogeneous_relation_with_torch_compressed_export():
    writes = ("author", "writes", "paper")
    graph = Graph.hetero(
        nodes={
            "author": {"x": torch.tensor([[1.0], [2.0]])},
            "paper": {"x": torch.tensor([[10.0], [20.0], [30.0]])},
        },
        edges={
            writes: {"edge_index": torch.tensor([[1, 0], [2, 1]])},
        },
    )

    csr = adj_external(graph, edge_type=writes, torch_fmt="csr")

    assert csr.layout is torch.sparse_csr
    assert tuple(csr.size()) == (2, 3)
    assert torch.equal(csr.crow_indices(), torch.tensor([0, 1, 2]))
    assert torch.equal(csr.col_indices(), torch.tensor([1, 2]))
    assert torch.equal(csr.values(), torch.tensor([1.0, 1.0]))


def test_adj_external_rejects_combined_torch_and_scipy_format_requests():
    graph = Graph.homo(
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        x=torch.tensor([[1.0], [2.0], [3.0]]),
    )

    with pytest.raises(ValueError, match="cannot both be set"):
        adj_external(graph, scipy_fmt="coo", torch_fmt="csr")
