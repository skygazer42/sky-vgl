from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sampling_guide_documents_prefetch_and_worker_boundaries():
    sampling_guide = (REPO_ROOT / "docs" / "guide" / "sampling.md").read_text(encoding="utf-8")

    assert "`prefetch` 只用于 `num_workers == 0` 的单线程路径" in sampling_guide
    assert "`prefetch_factor` 只用于 `num_workers > 0` 的 worker 预取" in sampling_guide
    assert "`persistent_workers=True` 也只适用于 `num_workers > 0`" in sampling_guide
    assert "不要在 `num_workers>0` 时设置 `prefetch`" in sampling_guide


def test_dataloading_api_documents_prefetch_and_worker_boundaries():
    api_docs = (REPO_ROOT / "docs" / "api" / "dataloading.md").read_text(encoding="utf-8")

    assert "`prefetch` 只用于 `num_workers == 0` 的单线程路径" in api_docs
    assert "`prefetch_factor` 只用于 `num_workers > 0`" in api_docs
    assert "`persistent_workers=True` 也只适用于 `num_workers > 0`" in api_docs
    assert "不要在 `num_workers>0` 时设置 `prefetch`" in api_docs


def test_dataloading_api_documents_prefetch_and_worker_boundaries():
    dataloading_api = (REPO_ROOT / "docs" / "api" / "dataloading.md").read_text(encoding="utf-8")

    assert "`prefetch` 只用于 `num_workers == 0` 的单线程路径" in dataloading_api
    assert "`prefetch_factor` 只用于 `num_workers > 0` 的 worker 预取" in dataloading_api
    assert "`persistent_workers=True` 也只适用于 `num_workers > 0`" in dataloading_api
    assert "当 `num_workers > 0` 时不要再设置 `prefetch`" in dataloading_api
