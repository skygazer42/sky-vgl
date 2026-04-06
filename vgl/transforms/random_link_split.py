import torch
from vgl.dataloading.dataset import ListDataset
from vgl.dataloading.records import LinkPredictionRecord
from vgl.graph.stores import EdgeStore
from vgl.graph.view import GraphView
from vgl.transforms.base import BaseTransform


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


def _sorted_unique_tensor(values) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    if values.numel() == 0:
        return values
    sorted_values = torch.sort(values, stable=True).values
    keep = torch.ones(sorted_values.numel(), dtype=torch.bool, device=sorted_values.device)
    if sorted_values.numel() > 1:
        keep[1:] = sorted_values[1:] != sorted_values[:-1]
    return sorted_values[keep]


def _membership_mask(values, allowed_values) -> torch.Tensor:
    values = torch.as_tensor(values, dtype=torch.long).view(-1)
    allowed_values = _sorted_unique_tensor(
        torch.as_tensor(allowed_values, dtype=torch.long, device=values.device).view(-1)
    )
    if values.numel() == 0 or allowed_values.numel() == 0:
        return torch.zeros(values.numel(), dtype=torch.bool, device=values.device)
    positions = torch.searchsorted(allowed_values, values, right=False)
    capped_positions = positions.clamp_max(allowed_values.numel() - 1)
    return (positions < allowed_values.numel()) & (allowed_values[capped_positions] == values)


def _edge_pair_keys(src_index, dst_index, *, dst_count: int) -> torch.Tensor:
    src_index = torch.as_tensor(src_index, dtype=torch.long).view(-1)
    dst_index = torch.as_tensor(dst_index, dtype=torch.long, device=src_index.device).view(-1)
    stride = torch.as_tensor(dst_count, dtype=torch.long, device=src_index.device).reshape(())
    return src_index * stride + dst_index


def _expand_interval_positions(starts: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    starts = torch.as_tensor(starts, dtype=torch.long).view(-1)
    counts = torch.as_tensor(counts, dtype=torch.long, device=starts.device).view(-1)
    if starts.numel() == 0 or counts.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=starts.device)

    positive = counts > 0
    starts = starts[positive]
    counts = counts[positive]
    if starts.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=starts.device)

    offsets = torch.cumsum(counts, dim=0) - counts
    bases = starts - offsets
    deltas = torch.empty_like(bases)
    deltas[0] = bases[0]
    if bases.numel() > 1:
        deltas[1:] = bases[1:] - bases[:-1]
    expanded = torch.zeros(counts.sum(), dtype=torch.long, device=starts.device)
    expanded[offsets] = deltas
    expanded = torch.cumsum(expanded, dim=0)
    expanded += torch.arange(counts.sum(), dtype=torch.long, device=starts.device)
    return expanded


def _merge_group_indices(groups, *, device: torch.device) -> torch.Tensor:
    if not groups:
        return torch.empty(0, dtype=torch.long, device=device)
    merged = torch.cat(
        [torch.as_tensor(group, dtype=torch.long, device=device).view(-1) for group in groups]
    )
    return torch.sort(merged, stable=True).values


def _slice_edge_store(store, indices):
    edge_count = int(store.edge_index.size(1))
    index = torch.as_tensor(indices, dtype=torch.long)
    edge_data = {}
    for key, value in store.data.items():
        if key == "edge_index":
            edge_data[key] = value[:, index]
        elif isinstance(value, torch.Tensor) and value.ndim > 0 and value.size(0) == edge_count:
            edge_data[key] = value[index]
        else:
            edge_data[key] = value
    return EdgeStore(store.type_name, edge_data)


def _edge_subgraph(graph, edge_indices_by_type):
    edges = dict(graph.edges)
    for edge_type, indices in edge_indices_by_type.items():
        edges[edge_type] = _slice_edge_store(graph.edges[edge_type], indices)
    base = getattr(graph, "base", graph)
    return GraphView(base=base, nodes=graph.nodes, edges=edges, schema=graph.schema)


class RandomLinkSplit(BaseTransform):
    def __init__(
        self,
        num_val=0.1,
        num_test=0.2,
        *,
        is_undirected=False,
        include_validation_edges_in_test=True,
        edge_type=None,
        rev_edge_type=None,
        disjoint_train_ratio=0.0,
        neg_sampling_ratio=0.0,
        add_negative_train_samples=False,
        seed=None,
    ):
        self.num_val = num_val
        self.num_test = num_test
        self.is_undirected = bool(is_undirected)
        self.include_validation_edges_in_test = bool(include_validation_edges_in_test)
        self.edge_type = None if edge_type is None else tuple(edge_type)
        self.rev_edge_type = None if rev_edge_type is None else tuple(rev_edge_type)
        self.disjoint_train_ratio = disjoint_train_ratio
        self.neg_sampling_ratio = neg_sampling_ratio
        self.add_negative_train_samples = bool(add_negative_train_samples)
        self.seed = seed

    def _resolve_count(self, value, total, name):
        if isinstance(value, float):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must be in [0.0, 1.0)")
            return int(total * value)
        count = _as_python_int(value)
        if count < 0:
            raise ValueError(f"{name} must be >= 0")
        if count >= total:
            raise ValueError(f"{name} must be smaller than the total number of edge groups")
        return count

    def _resolve_negative_count(self, num_positive):
        value = self.neg_sampling_ratio
        if isinstance(value, float):
            if value < 0.0:
                raise ValueError("neg_sampling_ratio must be >= 0")
            return int(num_positive * value)
        ratio = _as_python_int(value)
        if ratio < 0:
            raise ValueError("neg_sampling_ratio must be >= 0")
        return _as_python_int(num_positive) * ratio

    def _edge_groups(self, edge_index):
        num_edges = int(edge_index.size(1))
        if not self.is_undirected:
            return [torch.tensor([index], dtype=torch.long, device=edge_index.device) for index in range(num_edges)]
        if num_edges == 0:
            return []
        src_index = edge_index[0].to(dtype=torch.long)
        dst_index = edge_index[1].to(dtype=torch.long)
        normalized_src = torch.minimum(src_index, dst_index)
        normalized_dst = torch.maximum(src_index, dst_index)
        key_base = torch.maximum(normalized_src.max(), normalized_dst.max()) + 1
        edge_keys = _edge_pair_keys(normalized_src, normalized_dst, dst_count=key_base)
        order = torch.argsort(edge_keys, stable=True)
        sorted_keys = edge_keys.index_select(0, order)
        group_starts = torch.ones(sorted_keys.numel(), dtype=torch.bool, device=sorted_keys.device)
        if sorted_keys.numel() > 1:
            group_starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
        starts = torch.nonzero(group_starts, as_tuple=False).view(-1)
        ends = torch.cat((starts[1:], starts.new_tensor([order.numel()])))
        start_values = starts.detach().cpu().numpy().reshape(-1)
        end_values = ends.detach().cpu().numpy().reshape(-1)
        return [order[start:end] for start, end in zip(start_values, end_values)]

    def _resolve_edge_type(self, graph):
        if self.edge_type is not None:
            edge_type = self.edge_type
        elif len(graph.edges) == 1:
            edge_type = next(iter(graph.edges))
        else:
            try:
                edge_type = graph._default_edge_type()
            except AttributeError as exc:
                raise ValueError(
                    "RandomLinkSplit requires edge_type for heterogeneous graphs with multiple edge types"
                ) from exc
        if edge_type not in graph.edges:
            raise ValueError("RandomLinkSplit edge_type must exist in the source graph")
        return edge_type

    def _reverse_edge_indices(self, edge_index, reverse_edge_index, indices):
        query_indices = torch.as_tensor(indices, dtype=torch.long, device=edge_index.device).view(-1)
        if query_indices.numel() == 0 or reverse_edge_index.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=edge_index.device)

        reverse_src = reverse_edge_index[0].to(dtype=torch.long)
        reverse_dst = reverse_edge_index[1].to(dtype=torch.long)
        query_src = edge_index[1].index_select(0, query_indices).to(dtype=torch.long)
        query_dst = edge_index[0].index_select(0, query_indices).to(dtype=torch.long)
        key_base = torch.maximum(query_dst.max(), reverse_dst.max()) + 1

        reverse_keys = _edge_pair_keys(reverse_src, reverse_dst, dst_count=key_base)
        query_keys = _edge_pair_keys(query_src, query_dst, dst_count=key_base)
        order = torch.argsort(reverse_keys, stable=True)
        sorted_reverse_keys = reverse_keys.index_select(0, order)
        starts = torch.searchsorted(sorted_reverse_keys, query_keys, right=False)
        ends = torch.searchsorted(sorted_reverse_keys, query_keys, right=True)
        positions = _expand_interval_positions(starts, ends - starts)
        if positions.numel() == 0:
            return positions
        return _sorted_unique_tensor(order.index_select(0, positions))

    def _records_from_indices(self, graph, edge_index, indices, *, split, edge_type, reverse_edge_type):
        index = torch.as_tensor(indices, dtype=torch.long, device=edge_index.device).view(-1)
        if index.numel() == 0:
            return []
        selected_edge_index = edge_index.index_select(1, index)
        edge_ids = index.detach().cpu().numpy().reshape(-1)
        src_indices = selected_edge_index[0].detach().cpu().numpy().reshape(-1)
        dst_indices = selected_edge_index[1].detach().cpu().numpy().reshape(-1)
        records = []
        for edge_id, src_index, dst_index in zip(edge_ids, src_indices, dst_indices):
            edge_id_value = int(edge_id)
            src_index_value = int(src_index)
            dst_index_value = int(dst_index)
            sample_id = f"{split}:{edge_id_value}"
            records.append(
                LinkPredictionRecord(
                    graph=graph,
                    src_index=src_index_value,
                    dst_index=dst_index_value,
                    label=1,
                    metadata={
                        "split": split,
                        "edge_id": edge_id_value,
                        "edge_type": edge_type,
                        "reverse_edge_type": reverse_edge_type,
                        "sample_id": sample_id,
                        "query_id": sample_id,
                        "exclude_seed_edges": True,
                    },
                    sample_id=sample_id,
                    exclude_seed_edge=True,
                    edge_type=edge_type,
                    reverse_edge_type=reverse_edge_type,
                    query_id=sample_id,
                )
            )
        return records

    def _positive_edge_keys(self, edge_index, *, src_node_type, dst_node_type, num_dst_nodes):
        edge_keys = _edge_pair_keys(
            edge_index[0],
            edge_index[1],
            dst_count=num_dst_nodes,
        )
        if self.is_undirected and src_node_type == dst_node_type:
            edge_keys = torch.cat(
                (
                    edge_keys,
                    _edge_pair_keys(edge_index[1], edge_index[0], dst_count=num_dst_nodes),
                )
            )
        return _sorted_unique_tensor(edge_keys)

    def _sample_negative_edges(self, *, count, num_src_nodes, num_dst_nodes, excluded_edges, generator):
        if count <= 0:
            return []
        all_possible = num_src_nodes * num_dst_nodes
        excluded_edge_keys = _sorted_unique_tensor(excluded_edges)
        if all_possible <= int(excluded_edge_keys.numel()):
            raise ValueError("RandomLinkSplit could not sample negatives: no valid negative edges remain")

        sampled_edge_keys = torch.empty((0,), dtype=torch.long)
        while int(sampled_edge_keys.numel()) < count:
            remaining = count - int(sampled_edge_keys.numel())
            # Sample more than needed to reduce rejection loops on dense graphs.
            draw_count = max(remaining * 2, 32)
            src_index = torch.randint(num_src_nodes, (draw_count,), generator=generator)
            dst_index = torch.randint(num_dst_nodes, (draw_count,), generator=generator)
            sampled_keys = _edge_pair_keys(src_index, dst_index, dst_count=num_dst_nodes)
            invalid_mask = _membership_mask(sampled_keys, excluded_edge_keys)
            if sampled_edge_keys.numel() > 0:
                invalid_mask |= _membership_mask(sampled_keys, sampled_edge_keys)
            accepted = _sorted_unique_tensor(sampled_keys[~invalid_mask])[:remaining]
            if accepted.numel() == 0:
                continue
            sampled_edge_keys = torch.cat((sampled_edge_keys, accepted))

        sampled = []
        for edge_key in torch.sort(sampled_edge_keys[:count], stable=True).values:
            edge_key_value = _as_python_int(edge_key)
            sampled.append((edge_key_value // int(num_dst_nodes), edge_key_value % int(num_dst_nodes)))
        return sampled

    def _sample_negative_destinations_for_source(
        self,
        *,
        src_index,
        count,
        num_dst_nodes,
        excluded_edges,
        generator,
    ):
        if count <= 0:
            return []
        excluded_edge_keys = _sorted_unique_tensor(excluded_edges)
        destination_ids = torch.arange(num_dst_nodes, dtype=torch.long)
        query_keys = destination_ids + _as_python_int(src_index) * int(num_dst_nodes)
        candidates = destination_ids[~_membership_mask(query_keys, excluded_edge_keys)]
        if candidates.numel() == 0:
            raise ValueError("RandomLinkSplit could not sample negatives for a validation/test query")
        if count <= int(candidates.numel()):
            permutation = torch.randperm(int(candidates.numel()), generator=generator)[:count]
            return [
                int(dst_index)
                for dst_index in candidates.index_select(0, permutation).detach().cpu().numpy().reshape(-1)
            ]

        sampled = [int(dst_index) for dst_index in candidates.detach().cpu().numpy().reshape(-1)]
        remaining = count - int(candidates.numel())
        indices = torch.randint(int(candidates.numel()), (remaining,), generator=generator)
        sampled.extend(
            int(dst_index)
            for dst_index in candidates.index_select(0, indices).detach().cpu().numpy().reshape(-1)
        )
        return sampled

    def _attach_negative_records(
        self,
        *,
        graph,
        split,
        records,
        edge_type,
        reverse_edge_type,
        num_src_nodes,
        num_dst_nodes,
        excluded_edges,
        generator,
    ):
        if split == "train" and not self.add_negative_train_samples:
            return records
        negative_count = self._resolve_negative_count(len(records))
        if negative_count <= 0:
            return records
        positive_query_ids = [
            record.query_id if record.query_id is not None else record.metadata.get("query_id", record.sample_id)
            for record in records
        ]
        counts_by_record = [0] * len(records)
        for offset in range(negative_count):
            counts_by_record[offset % len(records)] += 1
        augmented = []
        sample_offset = 0
        for record, count in zip(records, counts_by_record):
            augmented.append(record)
            src_index = _as_python_int(record.src_index)
            destinations = self._sample_negative_destinations_for_source(
                src_index=src_index,
                count=count,
                num_dst_nodes=num_dst_nodes,
                excluded_edges=excluded_edges,
                generator=generator,
            )
            for dst_index in destinations:
                sample_id = f"{split}:neg:{sample_offset}"
                query_id = record.query_id if record.query_id is not None else positive_query_ids[sample_offset % len(positive_query_ids)]
                augmented.append(
                    LinkPredictionRecord(
                        graph=graph,
                        src_index=src_index,
                        dst_index=int(dst_index),
                        label=0,
                        metadata={
                            "split": split,
                            "edge_id": None,
                            "negative_sampled": True,
                            "edge_type": edge_type,
                            "reverse_edge_type": reverse_edge_type,
                            "sample_id": sample_id,
                            "query_id": query_id,
                        },
                        sample_id=sample_id,
                        edge_type=edge_type,
                        reverse_edge_type=reverse_edge_type,
                        query_id=query_id,
                    )
                )
                sample_offset += 1
        return augmented

    def __call__(self, graph):
        edge_type = self._resolve_edge_type(graph)
        edge_index = graph.edges[edge_type].edge_index
        src_node_type, _, dst_node_type = edge_type
        num_src_nodes = int(graph.nodes[src_node_type].x.size(0))
        num_dst_nodes = int(graph.nodes[dst_node_type].x.size(0))
        reverse_edge_index = None
        if self.rev_edge_type is not None:
            if self.rev_edge_type not in graph.edges:
                raise ValueError("RandomLinkSplit rev_edge_type must exist in the source graph")
            reverse_edge_index = graph.edges[self.rev_edge_type].edge_index

        groups = self._edge_groups(edge_index)
        total_groups = len(groups)
        if total_groups < 3:
            raise ValueError("RandomLinkSplit requires at least three edge groups")
        num_val = self._resolve_count(self.num_val, total_groups, "num_val")
        num_test = self._resolve_count(self.num_test, total_groups, "num_test")
        num_train = total_groups - num_val - num_test
        if num_train <= 0:
            raise ValueError("RandomLinkSplit requires at least one training edge group")

        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(_as_python_int(self.seed))
        permutation = torch.randperm(total_groups, generator=generator)
        shuffled_groups = [groups[index] for index in permutation.detach().cpu().numpy().reshape(-1)]

        train_group_indices = shuffled_groups[:num_train]
        val_group_indices = shuffled_groups[num_train:num_train + num_val]
        test_group_indices = shuffled_groups[num_train + num_val:]

        num_disjoint = self._resolve_count(
            self.disjoint_train_ratio,
            len(train_group_indices),
            "disjoint_train_ratio",
        ) if train_group_indices else 0
        if num_disjoint > 0:
            train_supervision_groups = train_group_indices[:num_disjoint]
            train_message_passing_groups = train_group_indices[num_disjoint:]
        else:
            train_supervision_groups = train_group_indices
            train_message_passing_groups = train_group_indices

        train_indices = _merge_group_indices(train_supervision_groups, device=edge_index.device)
        train_graph_indices = _merge_group_indices(train_message_passing_groups, device=edge_index.device)
        val_indices = _merge_group_indices(val_group_indices, device=edge_index.device)
        test_indices = _merge_group_indices(test_group_indices, device=edge_index.device)

        train_edge_indices = {edge_type: train_graph_indices}
        val_edge_indices = {edge_type: train_graph_indices}
        test_target_indices = (
            torch.cat((train_graph_indices, val_indices))
            if self.include_validation_edges_in_test
            else train_graph_indices
        )
        test_edge_indices = {edge_type: test_target_indices}
        if reverse_edge_index is not None:
            train_reverse_indices = self._reverse_edge_indices(edge_index, reverse_edge_index, train_graph_indices)
            val_reverse_indices = self._reverse_edge_indices(edge_index, reverse_edge_index, val_indices)
            train_val_reverse_indices = (
                torch.cat((train_reverse_indices, val_reverse_indices))
                if self.include_validation_edges_in_test
                else train_reverse_indices
            )
            train_edge_indices[self.rev_edge_type] = train_reverse_indices
            val_edge_indices[self.rev_edge_type] = train_reverse_indices
            test_edge_indices[self.rev_edge_type] = _sorted_unique_tensor(train_val_reverse_indices)

        train_graph = _edge_subgraph(graph, train_edge_indices)
        val_graph = _edge_subgraph(graph, val_edge_indices)
        if self.include_validation_edges_in_test:
            test_graph = _edge_subgraph(graph, test_edge_indices)
        else:
            test_graph = _edge_subgraph(graph, test_edge_indices)

        train_records = self._records_from_indices(
            train_graph,
            edge_index,
            train_indices,
            split="train",
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
        )
        val_records = self._records_from_indices(
            val_graph,
            edge_index,
            val_indices,
            split="val",
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
        )
        test_records = self._records_from_indices(
            test_graph,
            edge_index,
            test_indices,
            split="test",
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
        )

        generator = None
        if self.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(_as_python_int(self.seed) + 1)
        excluded_edges = self._positive_edge_keys(
            edge_index,
            src_node_type=src_node_type,
            dst_node_type=dst_node_type,
            num_dst_nodes=num_dst_nodes,
        )
        train_records = self._attach_negative_records(
            graph=train_graph,
            split="train",
            records=train_records,
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            excluded_edges=excluded_edges,
            generator=generator,
        )
        val_records = self._attach_negative_records(
            graph=val_graph,
            split="val",
            records=val_records,
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            excluded_edges=excluded_edges,
            generator=generator,
        )
        test_records = self._attach_negative_records(
            graph=test_graph,
            split="test",
            records=test_records,
            edge_type=edge_type,
            reverse_edge_type=self.rev_edge_type,
            num_src_nodes=num_src_nodes,
            num_dst_nodes=num_dst_nodes,
            excluded_edges=excluded_edges,
            generator=generator,
        )
        return ListDataset(train_records), ListDataset(val_records), ListDataset(test_records)
