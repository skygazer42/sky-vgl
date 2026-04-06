from __future__ import annotations

import torch

from vgl.transforms.base import BaseTransform
from vgl.transforms._utils import clone_graph


def _as_python_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.detach().cpu().numpy().reshape(()).item())
    return int(value)


class NormalizeFeatures(BaseTransform):
    def __init__(self, *, attr_name: str = "x", eps: float = 1e-12):
        self.attr_name = attr_name
        self.eps = float(eps)

    def __call__(self, graph):
        nodes = {
            node_type: dict(store.data)
            for node_type, store in graph.nodes.items()
        }
        for node_type, store in graph.nodes.items():
            features = store.data.get(self.attr_name)
            if not isinstance(features, torch.Tensor) or features.ndim < 2:
                continue
            denom = features.sum(dim=-1, keepdim=True).clamp_min(self.eps)
            nodes[node_type][self.attr_name] = features / denom
        return clone_graph(graph, nodes=nodes)


class FeatureStandardize(BaseTransform):
    def __init__(self, *, attr_name: str = "x", eps: float = 1e-12):
        self.attr_name = attr_name
        self.eps = float(eps)

    def _stats(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(self.eps)
        return mean, std

    def __call__(self, graph):
        nodes = {
            node_type: dict(store.data)
            for node_type, store in graph.nodes.items()
        }
        for node_type, store in graph.nodes.items():
            features = store.data.get(self.attr_name)
            if not isinstance(features, torch.Tensor) or features.ndim < 2:
                continue
            mean, std = self._stats(features)
            nodes[node_type][self.attr_name] = (features - mean) / std
        return clone_graph(graph, nodes=nodes)


class TrainOnlyFeatureNormalizer(FeatureStandardize):
    def __init__(self, *, attr_name: str = "x", train_mask_name: str = "train_mask", eps: float = 1e-12):
        super().__init__(attr_name=attr_name, eps=eps)
        self.train_mask_name = train_mask_name

    def __call__(self, graph):
        nodes = {
            node_type: dict(store.data)
            for node_type, store in graph.nodes.items()
        }
        for node_type, store in graph.nodes.items():
            features = store.data.get(self.attr_name)
            train_mask = store.data.get(self.train_mask_name)
            if not isinstance(features, torch.Tensor) or features.ndim < 2:
                continue
            if not isinstance(train_mask, torch.Tensor):
                raise ValueError(f"TrainOnlyFeatureNormalizer requires {self.train_mask_name!r}")
            train_mask = train_mask.to(dtype=torch.bool)
            if _as_python_int(train_mask.sum()) == 0:
                raise ValueError("TrainOnlyFeatureNormalizer requires at least one training node")
            mean, std = self._stats(features[train_mask])
            nodes[node_type][self.attr_name] = (features - mean) / std
        return clone_graph(graph, nodes=nodes)
