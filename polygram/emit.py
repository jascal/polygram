"""Public Q-Orca file emitter — `write_qorca(dictionary, path)`."""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
from pathlib import Path

from polygram._qorca_emit import render_machine_markdown
from polygram.dictionary import Dictionary


def _git_rev(repo_root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unversioned"


def _provenance_block(dictionary: Dictionary, repo_root: Path) -> str:
    ts = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rev = _git_rev(repo_root)
    return (
        f"<!--\n"
        f"  Polygram-generated artifact. Do not hand-edit.\n"
        f"  source dictionary: {dictionary.name}\n"
        f"  features:          {len(dictionary.features)} "
        f"in {len(dictionary.hierarchy)} clusters\n"
        f"  generated:         {ts}\n"
        f"  git rev:           {rev}\n"
        f"-->\n"
    )


def write_qorca(dictionary: Dictionary, path: str | os.PathLike) -> Path:
    """Write a `.q.orca.md` for `dictionary` at `path`. Returns the
    `Path` written. Includes a provenance comment block at the top of
    the file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = render_machine_markdown(dictionary)
    header = _provenance_block(dictionary, p.parent.resolve())
    p.write_text(header + body)
    return p
