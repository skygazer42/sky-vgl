PYTHON ?= python
BENCH_OUTPUT ?= benchmarks/latest.json
BENCH_PRESET ?= smoke
INTEROP_BACKEND ?= all
RELEASE_INTEROP_BACKEND ?= dgl

.PHONY: verify coverage metadata test lint types docs package full-scan docs-link public-surface release-contract release-smoke release-artifact-interop-smoke extras-smoke interop-smoke bench-smoke

verify: test coverage lint types docs package full-scan docs-link public-surface metadata release-contract release-smoke extras-smoke bench-smoke

coverage:
	$(PYTHON) -m pytest --cov=vgl --cov-report=term-missing --cov-fail-under=80 -q

metadata:
	$(PYTHON) scripts/metadata_consistency.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

types:
	$(PYTHON) -m mypy vgl

docs:
	$(PYTHON) -m mkdocs build --strict

package:
	$(PYTHON) -m build

full-scan:
	$(PYTHON) scripts/full_scan.py

docs-link:
	$(PYTHON) scripts/docs_link_scan.py

public-surface:
	$(PYTHON) scripts/public_surface_scan.py

release-contract:
	$(PYTHON) scripts/release_contract_scan.py --artifact-dir dist

release-smoke:
	$(PYTHON) scripts/release_smoke.py --artifact-dir dist --kind all

release-artifact-interop-smoke:
	$(PYTHON) scripts/release_smoke.py --artifact-dir dist --kind wheel --interop-backend=$(RELEASE_INTEROP_BACKEND)

extras-smoke:
	$(PYTHON) scripts/extras_smoke.py --extras networkx scipy tensorboard

interop-smoke:
	$(PYTHON) scripts/interop_smoke.py --backend=$(INTEROP_BACKEND)

bench-smoke:
	$(PYTHON) scripts/benchmark_hotpaths.py --preset=$(BENCH_PRESET) --output=$(BENCH_OUTPUT)
