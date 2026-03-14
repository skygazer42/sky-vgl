from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from gnn.core.graph import Graph

if TYPE_CHECKING:
    from gnn.data.sample import SampleRecord


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
