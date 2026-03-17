from vgl.dataloading.records import SampleRecord


class Sampler:
    def sample(self, item):
        raise NotImplementedError


class FullGraphSampler(Sampler):
    def sample(self, graph):
        return graph


class NodeSeedSubgraphSampler(Sampler):
    def sample(self, item):
        graph, metadata = item
        return SampleRecord(
            graph=graph,
            metadata=dict(metadata),
            sample_id=metadata.get("sample_id"),
            source_graph_id=metadata.get("source_graph_id"),
            subgraph_seed=metadata.get("seed"),
        )
