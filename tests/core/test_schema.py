import pytest

from gnn.core.schema import GraphSchema


def test_schema_tracks_node_edge_and_time_metadata():
    schema = GraphSchema(
        node_types=("paper", "author"),
        edge_types=(("author", "writes", "paper"),),
        node_features={"paper": ("x", "y"), "author": ("x",)},
        edge_features={("author", "writes", "paper"): ("timestamp",)},
        time_attr="timestamp",
    )

    assert schema.node_types == ("paper", "author")
    assert schema.edge_types == (("author", "writes", "paper"),)
    assert schema.time_attr == "timestamp"


def test_schema_rejects_unknown_time_field():
    with pytest.raises(ValueError, match="time_attr"):
        GraphSchema(
            node_types=("paper",),
            edge_types=(("paper", "cites", "paper"),),
            node_features={"paper": ("x",)},
            edge_features={("paper", "cites", "paper"): ("weight",)},
            time_attr="timestamp",
        )
