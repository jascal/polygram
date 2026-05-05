"""Shared sha256 helper for `Compressor` and `Regrower`.

Both write atomic checkpoints and need to hash their source + output
bytes for the audit trail in their respective reports.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of `path`'s bytes, streamed in 64KiB
    chunks (cheap on disk, doesn't materialize the file in memory)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
