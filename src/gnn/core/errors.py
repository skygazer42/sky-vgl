class GNNError(Exception):
    pass


class SchemaError(GNNError, ValueError):
    pass
