# Lazy On-Disk Dataset Design

## Goal

Turn `OnDiskGraphDataset` into a truly on-disk dataset that loads graphs lazily per item and exposes manifest-backed split views.

## Why This Batch

`OnDiskGraphDataset` now supports homogeneous, heterogeneous, and temporal graphs, but it still eagerly `torch.load(...)`s the entire `graphs.pt` payload during initialization. That means the API says “on-disk” while the implementation still behaves like an eager in-memory list. This is one of the clearest remaining data-ecosystem gaps for large datasets: a dataset container should not require loading every graph before the first sample is read. The existing `DatasetManifest.splits` metadata is also underused because split declarations do not yet translate into dataset views.

## Recommended Approach

Keep the public `OnDiskGraphDataset(root)` constructor stable, but change newly written datasets to store one graph payload per file under a deterministic `graphs/` directory. Then make the dataset object index those graph files lazily. The core approach is:

- `write(...)` emits `manifest.json` plus `graphs/graph-000000.pt`, `graphs/graph-000001.pt`, ...
- `OnDiskGraphDataset.__getitem__(index)` loads only the requested graph file and deserializes it
- `OnDiskGraphDataset.split(name)` returns a lightweight view over the relevant contiguous range described by the manifest
- if only legacy `graphs.pt` exists, the dataset should still load successfully through a fallback path

This gives new datasets real lazy loading without breaking older artifacts. It also keeps the batching and loader contracts unchanged because the dataset still behaves like a map-style dataset with `__len__` and `__getitem__`.

## Constraints

- Do not redesign `DatasetManifest` in this batch.
- Do not add random-access indices for non-contiguous split layouts; assume manifest splits describe contiguous ranges in order.
- Do not add multiprocessing-safe shared caches or mmap-backed graph payloads yet.
- Preserve current `serialize_graph(...)` / `deserialize_graph(...)` semantics.

## Testing Strategy

Use TDD in `tests/data/test_ondisk_dataset.py` and one integration path:

1. Add a failing regression test proving new writes create per-graph files and load lazily through item access.
2. Add a failing regression test proving `split(name)` returns a dataset view with the expected graphs and length.
3. Add a failing regression test proving legacy `graphs.pt` datasets still load.
4. Run the existing on-disk integration flow and then the full suite after docs are updated.

## Documentation Impact

Refresh `README.md`, `docs/core-concepts.md`, and `docs/quickstart.md` so `OnDiskGraphDataset` is described as a lazy per-graph on-disk container with manifest-backed split views, rather than an eager serialized graph list.
