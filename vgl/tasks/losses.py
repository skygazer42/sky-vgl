import torch
import torch.nn.functional as F


def normalize_class_weight(class_weight, *, device=None, dtype=None):
    if class_weight is None:
        return None
    weight = torch.as_tensor(class_weight, dtype=dtype or torch.float32, device=device)
    if weight.ndim != 1 or weight.numel() == 0:
        raise ValueError("class_weight must be a non-empty 1D tensor or sequence")
    return weight


def normalize_class_count(class_count, *, device=None, dtype=None):
    if class_count is None:
        return None
    count = torch.as_tensor(class_count, dtype=dtype or torch.float32, device=device)
    if count.ndim != 1 or count.numel() == 0:
        raise ValueError("class_count must be a non-empty 1D tensor or sequence")
    if torch.any(count <= 0):
        raise ValueError("class_count must be > 0")
    return count


def normalize_pos_weight(pos_weight, *, device=None, dtype=None):
    if pos_weight is None:
        return None
    weight = torch.as_tensor(pos_weight, dtype=dtype or torch.float32, device=device)
    if weight.ndim == 0:
        weight = weight.reshape(1)
    if weight.ndim != 1 or weight.numel() == 0:
        raise ValueError("pos_weight must be a positive scalar or non-empty 1D tensor or sequence")
    if torch.any(weight <= 0):
        raise ValueError("pos_weight must be > 0")
    return weight


def focal_cross_entropy(logits, targets, *, gamma=2.0, label_smoothing=0.0, class_weight=None):
    ce = F.cross_entropy(
        logits,
        targets,
        reduction="none",
        label_smoothing=label_smoothing,
        weight=class_weight,
    )
    pt = torch.softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (((1.0 - pt) ** gamma) * ce).mean()


def balanced_softmax_cross_entropy(logits, targets, *, class_count, label_smoothing=0.0):
    balanced_logits = logits + torch.log(class_count)
    return F.cross_entropy(
        balanced_logits,
        targets,
        label_smoothing=label_smoothing,
    )


def ldam_cross_entropy(
    logits,
    targets,
    *,
    class_count,
    max_margin=0.5,
    class_weight=None,
    label_smoothing=0.0,
):
    margins = torch.pow(class_count, -0.25)
    margins = margins * (max_margin / margins.max())
    adjusted_logits = logits.clone()
    adjusted_logits[torch.arange(targets.numel(), device=logits.device), targets] -= margins[targets]
    return F.cross_entropy(
        adjusted_logits,
        targets,
        weight=class_weight,
        label_smoothing=label_smoothing,
    )


def logit_adjusted_cross_entropy(
    logits,
    targets,
    *,
    class_count,
    tau=1.0,
    class_weight=None,
    label_smoothing=0.0,
):
    class_prior = class_count / class_count.sum()
    adjusted_logits = logits + tau * torch.log(class_prior)
    return F.cross_entropy(
        adjusted_logits,
        targets,
        weight=class_weight,
        label_smoothing=label_smoothing,
    )


def focal_binary_cross_entropy_with_logits(logits, targets, *, gamma=2.0, pos_weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    return (((1.0 - pt) ** gamma) * bce).mean()


__all__ = [
    "focal_binary_cross_entropy_with_logits",
    "focal_cross_entropy",
    "balanced_softmax_cross_entropy",
    "ldam_cross_entropy",
    "logit_adjusted_cross_entropy",
    "normalize_class_count",
    "normalize_class_weight",
    "normalize_pos_weight",
]
