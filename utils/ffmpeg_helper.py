from __future__ import annotations

import json
import subprocess
from pathlib import Path


def probe_media(path: str) -> dict[str, object]:
    media_path = Path(path)
    if not media_path.exists():
        raise FileNotFoundError(path)

    command = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(media_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout)

