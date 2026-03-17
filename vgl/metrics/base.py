class Metric:
    name = "metric"

    def reset(self):
        raise NotImplementedError

    def update(self, predictions, targets):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError
