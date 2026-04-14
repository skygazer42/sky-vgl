ARTIFACT_FORMAT_KEY = "format"
ARTIFACT_FORMAT_VERSION_KEY = "format_version"


def build_artifact_metadata(format_name: str, format_version: int, *, metadata=None) -> dict[str, object]:
    payload = {
        ARTIFACT_FORMAT_KEY: str(format_name),
        ARTIFACT_FORMAT_VERSION_KEY: int(format_version),
    }
    if metadata:
        for key, value in dict(metadata).items():
            if key in payload:
                raise ValueError(f"artifact metadata key {key!r} is reserved")
            payload[key] = value
    return payload
