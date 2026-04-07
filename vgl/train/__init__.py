"""Legacy compatibility namespace for training APIs.

New code should prefer ``vgl.engine``, ``vgl.tasks``, and ``vgl.metrics``.
"""

from vgl.engine import ASAM as ASAM
from vgl.engine import AdaptiveGradientClipping as AdaptiveGradientClipping
from vgl.engine import BootstrapBetaScheduler as BootstrapBetaScheduler
from vgl.engine import Callback as Callback
from vgl.engine import ConfidencePenaltyScheduler as ConfidencePenaltyScheduler
from vgl.engine import CHECKPOINT_FORMAT as CHECKPOINT_FORMAT
from vgl.engine import CHECKPOINT_FORMAT_VERSION as CHECKPOINT_FORMAT_VERSION
from vgl.engine import CSVLogger as CSVLogger
from vgl.engine import ConsoleLogger as ConsoleLogger
from vgl.engine import DeferredReweighting as DeferredReweighting
from vgl.engine import EarlyStopping as EarlyStopping
from vgl.engine import ExponentialMovingAverage as ExponentialMovingAverage
from vgl.engine import Evaluator as Evaluator
from vgl.engine import FocalGammaScheduler as FocalGammaScheduler
from vgl.engine import FloodingLevelScheduler as FloodingLevelScheduler
from vgl.engine import GeneralizedCrossEntropyScheduler as GeneralizedCrossEntropyScheduler
from vgl.engine import GradientAccumulationScheduler as GradientAccumulationScheduler
from vgl.engine import GradientNoiseInjection as GradientNoiseInjection
from vgl.engine import GradientValueClipping as GradientValueClipping
from vgl.engine import GradientCentralization as GradientCentralization
from vgl.engine import GSAM as GSAM
from vgl.engine import GradualUnfreezing as GradualUnfreezing
from vgl.engine import HistoryLogger as HistoryLogger
from vgl.engine import JSONLinesLogger as JSONLinesLogger
from vgl.engine import LabelSmoothingScheduler as LabelSmoothingScheduler
from vgl.engine import LdamMarginScheduler as LdamMarginScheduler
from vgl.engine import LogitAdjustTauScheduler as LogitAdjustTauScheduler
from vgl.engine import ModelCheckpoint as ModelCheckpoint
from vgl.engine import Logger as Logger
from vgl.engine import Poly1EpsilonScheduler as Poly1EpsilonScheduler
from vgl.engine import PosWeightScheduler as PosWeightScheduler
from vgl.engine import SymmetricCrossEntropyBetaScheduler as SymmetricCrossEntropyBetaScheduler
from vgl.engine import WeightDecayScheduler as WeightDecayScheduler
from vgl.engine import LayerwiseLrDecay as LayerwiseLrDecay
from vgl.engine import Lookahead as Lookahead
from vgl.engine import SAM as SAM
from vgl.engine import StochasticWeightAveraging as StochasticWeightAveraging
from vgl.engine import StopTraining as StopTraining
from vgl.engine import TensorBoardLogger as TensorBoardLogger
from vgl.engine import TrainingHistory as TrainingHistory
from vgl.engine import Trainer as Trainer
from vgl.engine import WarmupCosineScheduler as WarmupCosineScheduler
from vgl.engine import load_checkpoint as load_checkpoint
from vgl.engine import restore_checkpoint as restore_checkpoint
from vgl.engine import save_checkpoint as save_checkpoint
from vgl.metrics import Accuracy as Accuracy
from vgl.metrics import FilteredHitsAtK as FilteredHitsAtK
from vgl.metrics import FilteredMRR as FilteredMRR
from vgl.metrics import HitsAtK as HitsAtK
from vgl.metrics import Metric as Metric
from vgl.metrics import MRR as MRR
from vgl.metrics import build_metric as build_metric
from vgl.tasks import BootstrapTask as BootstrapTask
from vgl.tasks import ConfidencePenaltyTask as ConfidencePenaltyTask
from vgl.tasks import FloodingTask as FloodingTask
from vgl.tasks import GeneralizedCrossEntropyTask as GeneralizedCrossEntropyTask
from vgl.tasks import GraphClassificationTask as GraphClassificationTask
from vgl.tasks import LinkPredictionTask as LinkPredictionTask
from vgl.tasks import NodeClassificationTask as NodeClassificationTask
from vgl.tasks import Poly1CrossEntropyTask as Poly1CrossEntropyTask
from vgl.tasks import RDropTask as RDropTask
from vgl.tasks import SymmetricCrossEntropyTask as SymmetricCrossEntropyTask
from vgl.tasks import Task as Task
from vgl.tasks import TemporalEventPredictionTask as TemporalEventPredictionTask

__all__ = [
    "Accuracy",
    "ASAM",
    "AdaptiveGradientClipping",
    "BootstrapBetaScheduler",
    "BootstrapTask",
    "Callback",
    "ConfidencePenaltyScheduler",
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "CSVLogger",
    "ConsoleLogger",
    "DeferredReweighting",
    "EarlyStopping",
    "ExponentialMovingAverage",
    "Evaluator",
    "FilteredHitsAtK",
    "FilteredMRR",
    "FocalGammaScheduler",
    "FloodingLevelScheduler",
    "GeneralizedCrossEntropyScheduler",
    "SymmetricCrossEntropyBetaScheduler",
    "GradientAccumulationScheduler",
    "GradientNoiseInjection",
    "GradientValueClipping",
    "GradientCentralization",
    "GSAM",
    "GradualUnfreezing",
    "HistoryLogger",
    "HitsAtK",
    "JSONLinesLogger",
    "LabelSmoothingScheduler",
    "LdamMarginScheduler",
    "LogitAdjustTauScheduler",
    "ModelCheckpoint",
    "Logger",
    "Poly1EpsilonScheduler",
    "PosWeightScheduler",
    "WeightDecayScheduler",
    "LayerwiseLrDecay",
    "Lookahead",
    "Metric",
    "MRR",
    "SAM",
    "StochasticWeightAveraging",
    "StopTraining",
    "TensorBoardLogger",
    "TrainingHistory",
    "build_metric",
    "save_checkpoint",
    "load_checkpoint",
    "restore_checkpoint",
    "WarmupCosineScheduler",
    "ConfidencePenaltyTask",
    "FloodingTask",
    "GeneralizedCrossEntropyTask",
    "Task",
    "Trainer",
    "NodeClassificationTask",
    "GraphClassificationTask",
    "LinkPredictionTask",
    "Poly1CrossEntropyTask",
    "RDropTask",
    "SymmetricCrossEntropyTask",
    "TemporalEventPredictionTask",
]
