import inspect
import warnings


_warned_namespaces: set[str] = set()
_MIGRATION_GUIDE_SUFFIX = " See `docs/migration-guide.md` for import rewrite examples."


def warn_legacy_namespace(
    namespace: str,
    guidance: str,
    *,
    skip_when_imported_from: tuple[str, ...] = (),
) -> None:
    if namespace in _warned_namespaces:
        return
    if skip_when_imported_from:
        caller_filenames = {frame.filename.replace("\\", "/") for frame in inspect.stack(context=0)[1:8]}
        if any(filename.endswith(suffix) for filename in caller_filenames for suffix in skip_when_imported_from):
            return
    _warned_namespaces.add(namespace)
    warnings.warn(
        f"{namespace} is a legacy compatibility namespace; {guidance}{_MIGRATION_GUIDE_SUFFIX}",
        FutureWarning,
        stacklevel=2,
    )
