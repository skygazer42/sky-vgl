from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sampling_guide_documents_prefetch_and_worker_boundaries():
    sampling_guide = (REPO_ROOT / "docs" / "guide" / "sampling.md").read_text(encoding="utf-8")

    assert "`prefetch` 只用于 `num_workers == 0` 的单线程路径" in sampling_guide
    assert "`prefetch_factor` 只用于 `num_workers > 0` 的 worker 预取" in sampling_guide
    assert "`persistent_workers=True` 也只适用于 `num_workers > 0`" in sampling_guide
    assert "不要在 `num_workers>0` 时设置 `prefetch`" in sampling_guide
