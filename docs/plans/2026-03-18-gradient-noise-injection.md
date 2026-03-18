# Gradient Noise Injection Plan

## Goal

Add a callback-based training strategy that injects deterministic Gaussian noise into dense gradients before optimizer steps.

## Scope

- Add `GradientNoiseInjection` callback in `vgl.engine.callbacks`
- Support callback export from `vgl.engine`, `vgl.train`, `vgl.train.callbacks`, and `vgl`
- Add tests for:
  - callback configuration validation
  - deterministic gradient perturbation behavior
  - callback state restoration / checkpoint resume
  - package exports

## Design

- Add noise during `on_before_optimizer_step`, after backward and before optimizer step
- Scale noise as `std / step**decay_exponent`
- Use an internal CPU `torch.Generator` with saved state so pause/resume stays reproducible
- Skip sparse gradients and parameters without gradients

## Verification

- `python -m pytest tests/train/test_callbacks.py tests/train/test_trainer_evaluation.py tests/test_package_exports.py tests/test_package_layout.py -q`
- `python -m pytest -q`
- `python -m ruff check vgl tests examples`
