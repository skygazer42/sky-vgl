from pathlib import Path
from copy import deepcopy

import torch

from vgl._artifact import build_artifact_metadata


CHECKPOINT_FORMAT = "vgl.trainer_checkpoint"
CHECKPOINT_FORMAT_VERSION = 1


def build_checkpoint_payload(
    model_state_dict,
    *,
    metadata=None,
    optimizer_state_dict=None,
    lr_scheduler_state_dict=None,
    grad_scaler_state_dict=None,
    callback_states=None,
    trainer_state=None,
    history_state=None,
):
    payload = {
        **build_artifact_metadata(CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_VERSION),
        "model_state_dict": deepcopy(dict(model_state_dict)),
        "metadata": dict(metadata or {}),
    }
    optional_sections = {
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
        "callback_states": callback_states,
        "trainer_state": trainer_state,
        "history_state": history_state,
    }
    for key, value in optional_sections.items():
        if value is not None:
            payload[key] = deepcopy(value)
    return payload


def save_checkpoint(
    path,
    model_state_dict,
    *,
    metadata=None,
    optimizer_state_dict=None,
    lr_scheduler_state_dict=None,
    grad_scaler_state_dict=None,
    callback_states=None,
    trainer_state=None,
    history_state=None,
):
    checkpoint_path = Path(path)
    if checkpoint_path.exists() and checkpoint_path.is_dir():
        raise ValueError("checkpoint path must be a file path, not a directory")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        build_checkpoint_payload(
            model_state_dict,
            metadata=metadata,
            optimizer_state_dict=optimizer_state_dict,
            lr_scheduler_state_dict=lr_scheduler_state_dict,
            grad_scaler_state_dict=grad_scaler_state_dict,
            callback_states=callback_states,
            trainer_state=trainer_state,
            history_state=history_state,
        ),
        checkpoint_path,
    )
    return checkpoint_path


def checkpoint_event_fields(path, *, save_seconds):
    checkpoint_path = Path(path)
    return {
        **build_artifact_metadata(CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_VERSION),
        "path": str(checkpoint_path),
        "size_bytes": int(checkpoint_path.stat().st_size),
        "save_seconds": float(save_seconds),
    }


def normalize_checkpoint_payload(payload):
    if (
        isinstance(payload, dict)
        and payload.get("format") == CHECKPOINT_FORMAT
        and "model_state_dict" in payload
    ):
        normalized = {
            "format": payload["format"],
            "format_version": payload.get("format_version", CHECKPOINT_FORMAT_VERSION),
            "model_state_dict": payload["model_state_dict"],
            "metadata": dict(payload.get("metadata") or {}),
        }
        for key in (
            "optimizer_state_dict",
            "lr_scheduler_state_dict",
            "grad_scaler_state_dict",
            "callback_states",
            "trainer_state",
            "history_state",
        ):
            if key in payload:
                normalized[key] = payload[key]
        return normalized
    if isinstance(payload, dict):
        return {
            "format": "legacy.state_dict",
            "format_version": 0,
            "model_state_dict": payload,
            "metadata": {},
        }
    raise ValueError("Unsupported checkpoint format")


def load_checkpoint(path, *, map_location=None, weights_only=True):
    payload = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=weights_only,
    )
    return normalize_checkpoint_payload(payload)


def restore_checkpoint(model, path, *, map_location=None, strict=True, weights_only=True):
    payload = load_checkpoint(
        path,
        map_location=map_location,
        weights_only=weights_only,
    )
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    return payload
