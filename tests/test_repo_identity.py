from pathlib import Path


ACTIVE_TARGETS = [
    Path("README.md"),
    Path("docs/public-surface-contract.md"),
    Path("docs/core-concepts.md"),
    Path("docs/plans/2026-03-14-gnn-framework-design.md"),
    Path("docs/plans/2026-03-14-gnn-framework.md"),
    Path("docs/plans/2026-03-15-graph-classification-design.md"),
    Path("docs/plans/2026-03-15-graph-classification.md"),
    Path("docs/plans/2026-03-15-temporal-event-prediction-design.md"),
    Path("docs/plans/2026-03-15-temporal-event-prediction.md"),
]

BANNED_SNIPPETS = [
    'name = "g' + 'nn"',
    "from " + "gnn",
    "import " + "gnn",
    "src" + "/gnn",
]


def test_docs_and_config_use_vgl_identity():
    for path in ACTIVE_TARGETS:
        text = path.read_text(encoding="utf-8")
        for banned in BANNED_SNIPPETS:
            assert banned not in text, f"{path} still contains {banned!r}"
