from gnn.data.sample import SampleRecord


class FullGraphSampler:
    def sample(self, graph):
        return graph


class NodeSeedSubgraphSampler:
    def sample(self, item):
        graph, metadata = item
        return SampleRecord(
            graph=graph,
            metadata=dict(metadata),
            sample_id=metadata.get("sample_id"),
            source_graph_id=metadata.get("source_graph_id"),
            subgraph_seed=metadata.get("seed"),
        )
