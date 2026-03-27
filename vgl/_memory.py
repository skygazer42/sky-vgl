import torch


_PIN_MEMORY_UNAVAILABLE_MESSAGES = (
    "Found no NVIDIA driver",
    "Cannot access accelerator device when none is available",
)


def pin_tensor(tensor: torch.Tensor) -> torch.Tensor:
    try:
        return tensor.pin_memory()
    except RuntimeError as exc:
        if any(message in str(exc) for message in _PIN_MEMORY_UNAVAILABLE_MESSAGES):
            return tensor
        raise
