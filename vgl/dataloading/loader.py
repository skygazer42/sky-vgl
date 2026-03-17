from vgl.dataloading.records import LinkPredictionRecord, TemporalEventRecord
from vgl.graph.batch import GraphBatch, LinkPredictionBatch, TemporalEventBatch


class Loader:
    def __init__(self, dataset, sampler, batch_size, label_source=None, label_key=None):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.label_source = label_source
        self.label_key = label_key

    def _dataset_iter(self):
        try:
            return iter(self.dataset)
        except TypeError:
            if hasattr(self.dataset, "__len__") and hasattr(self.dataset, "__getitem__"):
                return (self.dataset[index] for index in range(len(self.dataset)))
            raise TypeError("Loader dataset must be iterable or implement __len__ and __getitem__")

    def _build_batch(self, items):
        if items and isinstance(items[0], TemporalEventRecord):
            return TemporalEventBatch.from_records(items)
        if items and isinstance(items[0], LinkPredictionRecord):
            return LinkPredictionBatch.from_records(items)
        if items and hasattr(items[0], "graph") and self.label_source is not None and self.label_key is not None:
            return GraphBatch.from_samples(
                items,
                label_key=self.label_key,
                label_source=self.label_source,
            )
        return GraphBatch.from_graphs(items)

    def __iter__(self):
        batch = []
        for item in self._dataset_iter():
            batch.append(self.sampler.sample(item))
            if len(batch) == self.batch_size:
                yield self._build_batch(batch)
                batch = []
        if batch:
            yield self._build_batch(batch)


DataLoader = Loader
