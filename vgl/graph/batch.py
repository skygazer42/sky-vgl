from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vgl.graph.graph import Graph

if TYPE_CHECKING:
    from vgl.dataloading.records import LinkPredictionRecord
    from vgl.dataloading.records import SampleRecord
    from vgl.dataloading.records import TemporalEventRecord


@dataclass(slots=True)
class GraphBatch:
    graphs: list[Graph]
    graph_index: torch.Tensor
    graph_ptr: torch.Tensor | None = None
    labels: torch.Tensor | None = None
    metadata: list[dict] | None = None

    @classmethod
    def from_graphs(cls, graphs: list[Graph]) -> "GraphBatch":
        counts = [graph.x.size(0) for graph in graphs]
        graph_index = torch.repeat_interleave(
            torch.arange(len(graphs)),
            torch.tensor(counts),
        )
        graph_ptr = torch.tensor([0, *torch.cumsum(torch.tensor(counts), dim=0).tolist()])
        return cls(graphs=graphs, graph_index=graph_index, graph_ptr=graph_ptr)

    @classmethod
    def from_samples(
        cls,
        samples: list["SampleRecord"],
        *,
        label_key: str,
        label_source: str,
    ) -> "GraphBatch":
        graphs = [sample.graph for sample in samples]
        batch = cls.from_graphs(graphs)
        batch.metadata = [sample.metadata for sample in samples]
        if label_source == "graph":
            batch.labels = torch.tensor([
                int(sample.graph.nodes["node"].data[label_key].reshape(-1)[0].item())
                for sample in samples
            ])
        elif label_source == "metadata":
            batch.labels = torch.tensor([int(sample.metadata[label_key]) for sample in samples])
        else:
            raise ValueError(f"Unsupported label_source: {label_source}")
        return batch

    @property
    def num_graphs(self) -> int:
        return len(self.graphs)


@dataclass(slots=True)
class LinkPredictionBatch:
    graph: Graph
    src_index: torch.Tensor
    dst_index: torch.Tensor
    labels: torch.Tensor
    metadata: list[dict] | None = None

    @classmethod
    def from_records(
        cls,
        records: list["LinkPredictionRecord"],
    ) -> "LinkPredictionBatch":
        if not records:
            raise ValueError("LinkPredictionBatch requires at least one record")
        graph = records[0].graph
        if any(record.graph is not graph for record in records):
            raise ValueError(
                "LinkPredictionBatch currently supports samples from a single source graph only"
            )

        src_index = torch.tensor([record.src_index for record in records], dtype=torch.long)
        dst_index = torch.tensor([record.dst_index for record in records], dtype=torch.long)
        labels = torch.tensor([float(record.label) for record in records], dtype=torch.float32)
        num_nodes = graph.x.size(0)

        if (
            (src_index < 0).any()
            or (src_index >= num_nodes).any()
            or (dst_index < 0).any()
            or (dst_index >= num_nodes).any()
        ):
            raise ValueError("LinkPredictionBatch indices must fall within the source graph node range")
        if not torch.all((labels == 0) | (labels == 1)):
            raise ValueError("LinkPredictionBatch labels must be binary 0/1")

        return cls(
            graph=graph,
            src_index=src_index,
            dst_index=dst_index,
            labels=labels,
            metadata=[record.metadata for record in records],
        )


@dataclass(slots=True)
class TemporalEventBatch:
    graph: Graph
    src_index: torch.Tensor
    dst_index: torch.Tensor
    timestamp: torch.Tensor
    labels: torch.Tensor
    event_features: torch.Tensor | None = None
    metadata: list[dict] | None = None

    @classmethod
    def from_records(
        cls,
        records: list["TemporalEventRecord"],
    ) -> "TemporalEventBatch":
        if not records:
            raise ValueError("TemporalEventBatch requires at least one record")
        graph = records[0].graph
        if graph.schema.time_attr is None:
            raise ValueError("TemporalEventBatch requires a temporal graph with schema.time_attr")
        if any(record.graph is not graph for record in records):
            raise ValueError(
                "TemporalEventBatch currently supports samples from a single source graph only"
            )
        event_features = [record.event_features for record in records]
        if all(feature is None for feature in event_features):
            stacked_event_features = None
        elif any(feature is None for feature in event_features):
            raise ValueError("TemporalEventBatch requires event_features for either all or none of the records")
        else:
            stacked_event_features = torch.stack([torch.as_tensor(feature) for feature in event_features], dim=0)
        return cls(
            graph=graph,
            src_index=torch.tensor([record.src_index for record in records]),
            dst_index=torch.tensor([record.dst_index for record in records]),
            timestamp=torch.tensor([record.timestamp for record in records]),
            labels=torch.tensor([record.label for record in records]),
            event_features=stacked_event_features,
            metadata=[record.metadata for record in records],
        )

    def history_graph(self, index: int):
        return self.graph.snapshot(self.timestamp[index].item())
