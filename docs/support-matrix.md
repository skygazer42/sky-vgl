---
hide:
  - navigation
  - toc
---

# Support Matrix

Track the combinations that CI and release verification currently exercise so you can reproduce a supported environment without guesswork.

## Core Languages & Runtimes

| Component | Verified Versions | Notes |
|-----------|-------------------|-------|
| Python | 3.10 / 3.11 / 3.12 | Ubuntu matrix plus a macOS smoke lane in CI. |
| PyTorch | 2.4+ | Core runtime dependency declared in packaging metadata and verified by the default CI install. |
| torch-geometric | 2.5+ (optional) | Manual/nightly `interop-smoke` installs the real extra and runs the reusable backend smoke script. |
| DGL | 2.1+ (optional) | Manual/nightly `interop-smoke` installs the real extra and runs the reusable backend smoke script. |

## Extras & Add-ons

| Extra | Purpose | Verification |
|-------|---------|--------------|
| `networkx` | Graph conversions ↔ NetworkX | Covered by CI `extras_smoke` and compatibility tests. |
| `scipy` | Sparse exports, algs | `extras_smoke` plus compatibility-oriented tests. |
| `tensorboard` | Logging | Verified via smoke script (Wave 1) and release packaging. |
| `dgl` | DGL interop | Manual/nightly `interop-smoke` installs the real extra and runs `python scripts/interop_smoke.py --backend dgl`. |
| `pyg` | PyG interop | Manual/nightly `interop-smoke` installs the real extra and runs `python scripts/interop_smoke.py --backend pyg`. |

## Installation Notes

1. Start from the Python version listed above.
2. Install the matching PyTorch wheel for your CUDA target (or CPU) before `pip install sky-vgl`.
3. Add extras via `pip install \"sky-vgl[networkx,scipy,...]\"` per the matrix row.
4. Run `python -m pytest tests/compat/test_optional_dependency_messages.py` after each upgrade to ensure metadata still advertises the extras.
5. For DGL/PyG environments, run `python scripts/interop_smoke.py --backend dgl` and/or `python scripts/interop_smoke.py --backend pyg` for the extras you installed. Use `--backend all` only when both are present.

## Documented Source of Truth

All of these expectations are driven from `scripts/contracts.py` so they stay in sync with release-item scans, tests, packaging metadata, and public surface verification.
