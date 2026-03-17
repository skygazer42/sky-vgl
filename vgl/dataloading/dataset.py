class ListDataset:
    def __init__(self, graphs):
        self.graphs = list(graphs)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]
