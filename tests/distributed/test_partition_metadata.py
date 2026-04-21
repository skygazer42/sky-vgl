import pytest

from vgl.distributed import (
    PartitionManifest,
    PartitionShard,
    load_partition_manifest,
    save_partition_manifest,
)


def test_partition_manifest_resolves_owner_and_counts_shards():
    manifest = PartitionManifest(
        num_nodes=6,
        partitions=(
            PartitionShard(partition_id=0, node_range=(0, 3), path="part-0.pt"),
            PartitionShard(partition_id=1, node_range=(3, 6), path="part-1.pt"),
        ),
        metadata={"name": "toy"},
    )

    assert manifest.num_partitions == 2
    assert manifest.owner(0).partition_id == 0
    assert manifest.owner(5).partition_id == 1


def test_partition_manifest_resolves_typed_node_ownership_and_ranges():
    manifest = PartitionManifest(
        num_nodes=9,
        num_nodes_by_type={"author": 4, "paper": 5},
        partitions=(
            PartitionShard(
                partition_id=0,
                node_range=(0, 2),
                node_ranges={"author": (0, 2), "paper": (0, 3)},
                path="part-0.pt",
            ),
            PartitionShard(
                partition_id=1,
                node_range=(2, 4),
                node_ranges={"author": (2, 4), "paper": (3, 5)},
                path="part-1.pt",
            ),
        ),
    )

    assert manifest.num_nodes_by_type == {"author": 4, "paper": 5}
    assert manifest.partitions[0].node_range_for("author") == (0, 2)
    assert manifest.partitions[0].node_range_for("paper") == (0, 3)
    assert manifest.owner(1, node_type="author").partition_id == 0
    assert manifest.owner(3, node_type="author").partition_id == 1
    assert manifest.owner(4, node_type="paper").partition_id == 1


def test_partition_manifest_validates_unique_ids_and_ranges():
    with pytest.raises(ValueError, match="unique"):
        PartitionManifest(
            num_nodes=4,
            partitions=(
                PartitionShard(partition_id=0, node_range=(0, 2)),
                PartitionShard(partition_id=0, node_range=(2, 4)),
            ),
        )

    with pytest.raises(ValueError, match="range"):
        PartitionManifest(
            num_nodes=4,
            partitions=(PartitionShard(partition_id=0, node_range=(0, 5)),),
        )


def test_partition_manifest_round_trips_edge_id_metadata(tmp_path):
    writes = ("author", "writes", "paper")
    manifest = PartitionManifest(
        num_nodes=9,
        num_nodes_by_type={"author": 4, "paper": 5},
        partitions=(
            PartitionShard(
                partition_id=0,
                node_range=(0, 2),
                node_ranges={"author": (0, 2), "paper": (0, 3)},
                path="part-0.pt",
                metadata={
                    "edge_ids_by_type": {
                        writes: [0, 1],
                    },
                    "boundary_edge_ids_by_type": {
                        writes: [4],
                    },
                },
            ),
            PartitionShard(
                partition_id=1,
                node_range=(2, 4),
                node_ranges={"author": (2, 4), "paper": (3, 5)},
                path="part-1.pt",
                metadata={
                    "edge_ids_by_type": {
                        writes: [2, 3],
                    },
                    "boundary_edge_ids_by_type": {
                        writes: [4],
                    },
                },
            ),
        ),
    )

    save_partition_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_partition_manifest(tmp_path / "manifest.json")

    assert loaded.partitions[0].metadata["edge_ids_by_type"][writes] == (0, 1)
    assert loaded.partitions[0].metadata["boundary_edge_ids_by_type"][writes] == (4,)
    assert loaded.partitions[1].metadata["edge_ids_by_type"][writes] == (2, 3)
    assert loaded.partitions[1].metadata["boundary_edge_ids_by_type"][writes] == (4,)


def test_partition_manifest_round_trips_feature_shape_metadata(tmp_path):
    writes = ("author", "writes", "paper")
    manifest = PartitionManifest(
        num_nodes=9,
        num_nodes_by_type={"author": 4, "paper": 5},
        partitions=(
            PartitionShard(
                partition_id=0,
                node_range=(0, 2),
                node_ranges={"author": (0, 2), "paper": (0, 3)},
                path="part-0.pt",
                metadata={
                    "node_feature_shapes": {
                        "author": {
                            "x": [2, 8],
                            "n_id": [2],
                        },
                    },
                    "node_feature_dtypes": {
                        "author": {
                            "x": "float32",
                            "n_id": "int64",
                        },
                    },
                    "edge_feature_shapes": {
                        writes: {
                            "weight": [2],
                            "e_id": [2],
                        },
                    },
                    "edge_feature_dtypes": {
                        writes: {
                            "weight": "float32",
                            "e_id": "int64",
                        },
                    },
                },
            ),
        ),
    )

    save_partition_manifest(tmp_path / "manifest.json", manifest)
    loaded = load_partition_manifest(tmp_path / "manifest.json")

    assert loaded.partitions[0].metadata["node_feature_shapes"]["author"]["x"] == (2, 8)
    assert loaded.partitions[0].metadata["node_feature_shapes"]["author"]["n_id"] == (2,)
    assert loaded.partitions[0].metadata["node_feature_dtypes"]["author"]["x"] == "float32"
    assert loaded.partitions[0].metadata["node_feature_dtypes"]["author"]["n_id"] == "int64"
    assert loaded.partitions[0].metadata["edge_feature_shapes"][writes]["weight"] == (2,)
    assert loaded.partitions[0].metadata["edge_feature_shapes"][writes]["e_id"] == (2,)
    assert loaded.partitions[0].metadata["edge_feature_dtypes"][writes]["weight"] == "float32"
    assert loaded.partitions[0].metadata["edge_feature_dtypes"][writes]["e_id"] == "int64"
