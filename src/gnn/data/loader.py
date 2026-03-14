from gnn.core.batch import GraphBatch


class Loader:
    def __init__(self, dataset, sampler, batch_size):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for graph in self.dataset.graphs:
            batch.append(self.sampler.sample(graph))
            if len(batch) == self.batch_size:
                yield GraphBatch.from_graphs(batch)
                batch = []
        if batch:
            yield GraphBatch.from_graphs(batch)
