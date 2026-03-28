# Full Repo Scan 100 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build and run one repeatable repository-wide scan surface that covers 100 concrete contract checks for packaging, docs, workflows, examples, assets, package layout, and regression anchors.

**Architecture:** Add one stdlib-first scan runner under `scripts/full_scan.py` that assembles a declarative 100-task catalog and evaluates each task against the live checkout. Lock the catalog with dedicated tests and wire the runner into CI so the scan remains executable and does not drift into stale documentation.

**Tech Stack:** Python 3.10+, `tomli` / `tomllib`, pytest, GitHub Actions

---

## Files

- Create: `scripts/full_scan.py`
- Create: `tests/test_full_scan.py`
- Modify: `.github/workflows/ci.yml`
- Create: `docs/plans/2026-03-28-full-repo-scan-100.md`

## Scan Catalog

### Tasks 1-10: Repository Surface

1. Scan `README.md` exists.
2. Scan `LICENSE` exists.
3. Scan `pyproject.toml` exists.
4. Scan `vgl/version.py` exists.
5. Scan `docs/core-concepts.md` exists.
6. Scan `docs/quickstart.md` exists.
7. Scan `docs/releasing.md` exists.
8. Scan `.github/workflows/ci.yml` exists.
9. Scan `.github/workflows/publish.yml` exists.
10. Scan `scripts/release_smoke.py` exists.

### Tasks 11-25: Packaging Metadata

11. Scan project name is `sky-vgl`.
12. Scan `requires-python` is `>=3.10`.
13. Scan runtime dependency `torch>=2.4` exists.
14. Scan runtime dependency `typing_extensions>=4.12` exists.
15. Scan Homepage URL is set.
16. Scan Repository URL is set.
17. Scan Documentation URL is set.
18. Scan Issues URL is set.
19. Scan Changelog URL is set.
20. Scan build backend is `hatchling.build`.
21. Scan version source path is `vgl/version.py`.
22. Scan wheel package target is `vgl`.
23. Scan sdist excludes `docs/plans`.
24. Scan sdist includes `docs/releasing.md`.
25. Scan sdist includes `scripts/release_smoke.py`.

### Tasks 26-35: Optional Dependency Surface

26. Scan optional extra `dev` exists.
27. Scan optional extra `scipy` exists.
28. Scan optional extra `networkx` exists.
29. Scan optional extra `tensorboard` exists.
30. Scan optional extra `dgl` exists.
31. Scan optional extra `pyg` exists.
32. Scan optional extra `full` exists.
33. Scan `dev` extra includes `pytest`.
34. Scan `dev` extra includes `ruff`.
35. Scan `dev` extra includes `mypy`.

### Tasks 36-45: README Install and Branding Contract

36. Scan `README.md` documents `pip install sky-vgl`.
37. Scan `README.md` documents `sky-vgl[full]`.
38. Scan `README.md` documents `sky-vgl[networkx]`.
39. Scan `README.md` documents `sky-vgl[pyg]`.
40. Scan `README.md` documents `sky-vgl[dgl]`.
41. Scan `README.md` documents the GitHub clone URL.
42. Scan `README.md` documents `cd sky-vgl`.
43. Scan `README.md` uses public raw URL for `logo.svg`.
44. Scan `README.md` uses public raw URL for `graph-types.svg`.
45. Scan `README.md` uses public raw URL for `architecture.svg`.

### Tasks 46-55: Quickstart and Release Docs Contract

46. Scan `docs/quickstart.md` documents `pip install sky-vgl`.
47. Scan `docs/quickstart.md` documents `sky-vgl[full]`.
48. Scan `docs/quickstart.md` explains that imports remain under `vgl`.
49. Scan `docs/releasing.md` includes `python -m build`.
50. Scan `docs/releasing.md` includes `python -m twine check`.
51. Scan `docs/releasing.md` includes `python scripts/release_smoke.py --artifact-dir dist --kind all`.
52. Scan `docs/releasing.md` mentions TestPyPI verification.
53. Scan `docs/releasing.md` mentions pending publisher handling.
54. Scan `docs/releasing.md` mentions `Manage Project -> Publishing`.
55. Scan `docs/releasing.md` mentions `PYPI_API_TOKEN`.

### Tasks 56-65: CI Workflow Contract

56. Scan CI workflow triggers on pushes to `main`.
57. Scan CI workflow triggers on pull requests.
58. Scan CI workflow defines the `test` job.
59. Scan CI workflow tests Python `3.10`.
60. Scan CI workflow tests Python `3.11`.
61. Scan CI workflow runs `python -m pytest -q`.
62. Scan CI workflow defines the `package-check` job.
63. Scan CI workflow runs `python -m build`.
64. Scan CI workflow runs `python -m twine check`.
65. Scan CI workflow runs `python scripts/release_smoke.py --artifact-dir dist --kind all`.

### Tasks 66-75: Publish Workflow Contract

66. Scan publish workflow triggers on `v*` tags.
67. Scan publish workflow triggers on published releases.
68. Scan publish workflow supports manual dispatch.
69. Scan publish workflow defines the auth probe job.
70. Scan publish workflow defines the build job.
71. Scan publish workflow has a TestPyPI API-token path.
72. Scan publish workflow has a TestPyPI trusted-publishing path.
73. Scan publish workflow has a PyPI API-token path.
74. Scan publish workflow has a PyPI trusted-publishing path.
75. Scan publish workflow uploads built distributions as artifacts.

### Tasks 76-85: Assets and Examples Surface

76. Scan `assets/logo.svg` exists.
77. Scan `assets/graph-types.svg` exists.
78. Scan `assets/architecture.svg` exists.
79. Scan `assets/pipeline.svg` exists.
80. Scan `assets/conv-layers.svg` exists.
81. Scan `examples/homo/node_classification.py` exists.
82. Scan `examples/homo/link_prediction.py` exists.
83. Scan `examples/hetero/node_classification.py` exists.
84. Scan `examples/hetero/link_prediction.py` exists.
85. Scan `examples/temporal/event_prediction.py` exists.

### Tasks 86-95: Package Layout Surface

86. Scan `vgl/graph` package exists.
87. Scan `vgl/dataloading` package exists.
88. Scan `vgl/storage` package exists.
89. Scan `vgl/sparse` package exists.
90. Scan `vgl/ops` package exists.
91. Scan `vgl/distributed` package exists.
92. Scan `vgl/nn` package exists.
93. Scan `vgl/tasks` package exists.
94. Scan `vgl/engine` package exists.
95. Scan `vgl/compat` package exists.

### Tasks 96-100: Regression Anchors

96. Scan `tests/test_release_packaging.py` exists.
97. Scan `tests/test_runtime_compat.py` exists.
98. Scan `tests/test_package_layout.py` exists.
99. Scan `tests/core/test_graph_ops_api.py` exists.
100. Scan `tests/data/test_loader.py` exists.

## Execution Notes

- Write `tests/test_full_scan.py` first and run it to observe the missing-script failure.
- Implement the scan runner with one declarative 100-task catalog and one CLI.
- Run the targeted scan tests until they pass.
- Wire the scan runner into CI as its own job or explicit step.
- Re-run the targeted scan tests plus the full test suite before merging.
