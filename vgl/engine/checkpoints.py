from pathlib import Path
from copy import deepcopy

import torch

from vgl._artifact import (
    ARTIFACT_FORMAT_KEY,
    ARTIFACT_FORMAT_VERSION_KEY,
    build_artifact_metadata,
    read_artifact_metadata,
)


CHECKPOINT_FORMAT = "vgl.trainer_checkpoint"
CHECKPOINT_FORMAT_VERSION = 1
LEGACY_CHECKPOINT_FORMAT = "legacy.state_dict"
LEGACY_CHECKPOINT_FORMAT_VERSION = 0


def _ensure_mapping_section(payload, key):
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be a mapping")
    return value


def _ensure_callback_states(payload):
    value = payload.get("callback_states")
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("callback_states must be a sequence")
    normalized = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError("callback_states entries must be mappings")
        callback_name = entry.get("callback")
        if not isinstance(callback_name, str):
            raise ValueError("callback_states entry callback must be a string")
        try:
            callback_index = int(entry.get("index", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("callback_states entry index must be an integer") from exc
        if callback_index < 0:
            raise ValueError("callback_states entry index must be >= 0")
        state = entry.get("state")
        if state is not None and not isinstance(state, dict):
            raise ValueError("callback_states entry state must be a mapping")
        normalized.append(
            {
                **entry,
                "callback": callback_name,
                "index": callback_index,
            }
        )
    return normalized


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
    if not isinstance(model_state_dict, dict):
        raise ValueError("model_state_dict must be a mapping")
    payload = {
        **build_artifact_metadata(CHECKPOINT_FORMAT, CHECKPOINT_FORMAT_VERSION),
        "model_state_dict": deepcopy(dict(model_state_dict)),
        "metadata": {} if metadata is None else deepcopy(metadata),
    }
    optional_sections = {
        "optimizer_state_dict": optimizer_state_dict,
        "lr_scheduler_state_dict": lr_scheduler_state_dict,
        "grad_scaler_state_dict": grad_scaler_state_dict,
        "callback_states": callback_states,
        "trainer_state": trainer_state,
        "history_state": history_state,
    }
    if trainer_state is not None:
        from vgl.engine.trainer import _normalize_restored_trainer_state

        optional_sections["trainer_state"] = _normalize_restored_trainer_state(trainer_state)
    if trainer_state is not None and history_state is not None:
        from vgl.engine.history import TrainingHistory

        normalized_history_state = TrainingHistory.from_state_dict(history_state).state_dict()
        optional_sections["history_state"] = normalized_history_state
        completed_epochs = int(normalized_history_state.get("completed_epochs", 0))
        trainer_global_step = optional_sections["trainer_state"].get("global_step")
        if trainer_global_step is None and completed_epochs > 0:
            raise ValueError("trainer_state.global_step must be >= history_state.completed_epochs")
        if trainer_global_step is not None and trainer_global_step < completed_epochs:
            raise ValueError("trainer_state.global_step must be >= history_state.completed_epochs")
        trainer_best_epoch = optional_sections["trainer_state"].get("best_epoch")
        history_best_epoch = normalized_history_state.get("best_epoch")
        if (
            trainer_best_epoch is not None
            and history_best_epoch is not None
            and trainer_best_epoch != history_best_epoch
        ):
            raise ValueError("trainer_state.best_epoch must match history_state.best_epoch")
        trainer_best_metric = optional_sections["trainer_state"].get("best_metric")
        history_best_metric = normalized_history_state.get("best_metric")
        if (
            trainer_best_metric is not None
            and history_best_metric is not None
            and trainer_best_metric != history_best_metric
        ):
            raise ValueError("trainer_state.best_metric must match history_state.best_metric")
    for key, value in optional_sections.items():
        if value is not None:
            payload[key] = deepcopy(value)
    return normalize_checkpoint_payload(payload)


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
    checkpoint_format, checkpoint_format_version = read_artifact_metadata(
        payload,
        default_format=None,
        default_format_version=CHECKPOINT_FORMAT_VERSION,
    ) if isinstance(payload, dict) else (None, None)
    if (
        isinstance(payload, dict)
        and checkpoint_format == CHECKPOINT_FORMAT
        and "model_state_dict" in payload
    ):
        metadata = payload.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a mapping")
        normalized = {
            ARTIFACT_FORMAT_KEY: checkpoint_format,
            ARTIFACT_FORMAT_VERSION_KEY: checkpoint_format_version,
            "metadata": dict(metadata),
        }
        model_state_dict = _ensure_mapping_section(payload, "model_state_dict")
        if model_state_dict is None:
            raise ValueError("model_state_dict must be a mapping")
        normalized["model_state_dict"] = model_state_dict
        for key in (
            "optimizer_state_dict",
            "lr_scheduler_state_dict",
            "grad_scaler_state_dict",
            "trainer_state",
            "history_state",
        ):
            value = _ensure_mapping_section(payload, key)
            if value is not None:
                normalized[key] = value
        callback_states = _ensure_callback_states(payload)
        if callback_states is not None:
            normalized["callback_states"] = callback_states
        history_state = normalized.get("history_state")
        if history_state is not None:
            from vgl.engine.history import TrainingHistory

            normalized["history_state"] = TrainingHistory.from_state_dict(history_state).state_dict()
        return normalized
    if isinstance(payload, dict):
        return {
            **build_artifact_metadata(LEGACY_CHECKPOINT_FORMAT, LEGACY_CHECKPOINT_FORMAT_VERSION),
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
