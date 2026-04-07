from __future__ import annotations

import email
import zipfile
from pathlib import Path


def read_wheel_metadata(wheel_path: Path) -> tuple[email.message.Message | None, str]:
    with zipfile.ZipFile(wheel_path) as archive:
        metadata_name = next((name for name in archive.namelist() if name.endswith("METADATA")), None)
        if metadata_name is None:
            return None, "wheel METADATA missing"
        return email.message_from_bytes(archive.read(metadata_name)), metadata_name
