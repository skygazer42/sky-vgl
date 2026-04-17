class GNNError(Exception):
    pass


class GraphConstructionError(GNNError, ValueError):
    pass


class SchemaError(GNNError, ValueError):
    pass
