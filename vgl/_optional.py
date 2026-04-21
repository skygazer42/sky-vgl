import importlib

_DIST_NAME = "sky-vgl"


def import_optional(
    module_name: str,
    *,
    package_name: str | None = None,
    extra_name: str,
    feature_name: str,
):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        expected_names = {
            module_name,
            module_name.split(".")[0],
        }
        install_name = package_name or module_name
        expected_names.add(install_name)
        expected_names.add(install_name.replace("-", "_"))
        missing_name = getattr(exc, "name", None)
        if missing_name is not None and missing_name not in expected_names:
            raise
        raise ImportError(
            f"{feature_name} requires the optional {install_name!r} dependency. "
            f'Install it with `pip install "{_DIST_NAME}[{extra_name}]"`.'
        ) from exc
