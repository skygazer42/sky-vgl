"""Public dataset namespace with dataloading compatibility re-exports.

Preferred imports:
- dataset and catalog APIs stay under ``vgl.data``
- loaders, samplers, plans, and materialization helpers should use ``vgl.dataloading``
"""
from vgl.dataloading import CandidateLinkSampler as CandidateLinkSampler
from vgl.dataloading import ClusterData as ClusterData
from vgl.dataloading import ClusterLoader as ClusterLoader
from vgl.dataloading import DataLoader as DataLoader
from vgl.dataloading import FullGraphSampler as FullGraphSampler
from vgl.dataloading import GraphSAINTEdgeSampler as GraphSAINTEdgeSampler
from vgl.dataloading import GraphSAINTNodeSampler as GraphSAINTNodeSampler
from vgl.dataloading import GraphSAINTRandomWalkSampler as GraphSAINTRandomWalkSampler
from vgl.dataloading import HardNegativeLinkSampler as HardNegativeLinkSampler
from vgl.dataloading import LinkNeighborSampler as LinkNeighborSampler
from vgl.dataloading import LinkPredictionRecord as LinkPredictionRecord
from vgl.dataloading import LinkSeedRequest as LinkSeedRequest
from vgl.dataloading import ListDataset as ListDataset
from vgl.dataloading import Loader as Loader
from vgl.dataloading import MaterializationContext as MaterializationContext
from vgl.dataloading import Node2VecWalkSampler as Node2VecWalkSampler
from vgl.dataloading import NodeNeighborSampler as NodeNeighborSampler
from vgl.dataloading import NodeSeedRequest as NodeSeedRequest
from vgl.dataloading import NodeSeedSubgraphSampler as NodeSeedSubgraphSampler
from vgl.dataloading import PlanExecutor as PlanExecutor
from vgl.dataloading import PlanStage as PlanStage
from vgl.dataloading import RandomWalkSampler as RandomWalkSampler
from vgl.dataloading import SampleRecord as SampleRecord
from vgl.dataloading import Sampler as Sampler
from vgl.dataloading import SamplingPlan as SamplingPlan
from vgl.dataloading import ShaDowKHopSampler as ShaDowKHopSampler
from vgl.dataloading import TemporalEventRecord as TemporalEventRecord
from vgl.dataloading import TemporalNeighborSampler as TemporalNeighborSampler
from vgl.dataloading import TemporalSeedRequest as TemporalSeedRequest
from vgl.dataloading import UniformNegativeLinkSampler as UniformNegativeLinkSampler
from vgl.dataloading import GraphSeedRequest as GraphSeedRequest
from vgl.dataloading import materialize_batch as materialize_batch
from vgl.dataloading import materialize_context as materialize_context
from vgl.data.cache import CACHE_ENV_VAR as CACHE_ENV_VAR
from vgl.data.cache import DataCache as DataCache
from vgl.data.cache import fingerprint_manifest as fingerprint_manifest
from vgl.data.cache import resolve_cache_dir as resolve_cache_dir
from vgl.data.catalog import DatasetCatalog as DatasetCatalog
from vgl.data.catalog import DatasetManifest as DatasetManifest
from vgl.data.catalog import DatasetSplit as DatasetSplit
from vgl.data.datasets import BuiltinDataset as BuiltinDataset
from vgl.data.datasets import ToyGraphDataset as ToyGraphDataset
from vgl.data.ondisk import OnDiskGraphDataset as OnDiskGraphDataset
from vgl.data.public import DatasetRegistry as DatasetRegistry
from vgl.data.public import KarateClubDataset as KarateClubDataset
from vgl.data.public import PlanetoidDataset as PlanetoidDataset
from vgl.data.public import TUDataset as TUDataset

__all__ = [
    "CACHE_ENV_VAR",
    "DataCache",
    "DataLoader",
    "DatasetCatalog",
    "DatasetManifest",
    "DatasetSplit",
    "DatasetRegistry",
    "KarateClubDataset",
    "OnDiskGraphDataset",
    "ListDataset",
    "Loader",
    "PlanetoidDataset",
    "Sampler",
    "TUDataset",
    "BuiltinDataset",
    "ToyGraphDataset",
    "CandidateLinkSampler",
    "ClusterData",
    "ClusterLoader",
    "FullGraphSampler",
    "GraphSAINTEdgeSampler",
    "GraphSAINTNodeSampler",
    "GraphSAINTRandomWalkSampler",
    "GraphSeedRequest",
    "HardNegativeLinkSampler",
    "LinkNeighborSampler",
    "NodeNeighborSampler",
    "LinkSeedRequest",
    "NodeSeedSubgraphSampler",
    "Node2VecWalkSampler",
    "NodeSeedRequest",
    "RandomWalkSampler",
    "ShaDowKHopSampler",
    "TemporalNeighborSampler",
    "TemporalSeedRequest",
    "UniformNegativeLinkSampler",
    "LinkPredictionRecord",
    "MaterializationContext",
    "PlanExecutor",
    "PlanStage",
    "SampleRecord",
    "SamplingPlan",
    "TemporalEventRecord",
    "fingerprint_manifest",
    "materialize_batch",
    "materialize_context",
    "resolve_cache_dir",
]
