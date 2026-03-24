import pytest

from vgl.distributed import PartitionManifest, PartitionShard


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
