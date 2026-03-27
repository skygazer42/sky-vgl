import importlib


def import_optional(
    module_name: str,
    *,
    package_name: str | None = None,
    extra_name: str,
    feature_name: str,
):
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        install_name = package_name or module_name
        raise ImportError(
            f"{feature_name} requires the optional {install_name!r} dependency. "
            f'Install it with `pip install "vgl[{extra_name}]"`.'
        ) from exc
