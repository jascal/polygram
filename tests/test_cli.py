"""Tests for the `polygram` console script."""

from __future__ import annotations

from pathlib import Path

import pytest

from polygram.cli import main


def _write_target(tmp_path: Path, body: str) -> Path:
    target = tmp_path / "myexample.py"
    target.write_text(body)
    return target


def test_run_target_writes_to_output_dir(tmp_path, capsys):
    target = _write_target(
        tmp_path,
        "from pathlib import Path\n"
        "def main(output_dir):\n"
        "    Path(output_dir).joinpath('hello').write_text('hi')\n",
    )
    out = tmp_path / "out"
    rc = main(["run", str(target), "--output-dir", str(out)])
    assert rc == 0
    assert (out / "hello").read_text() == "hi"


def test_run_target_missing_main_errors(tmp_path):
    target = _write_target(tmp_path, "x = 1\n")
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="no `main"):
        main(["run", str(target), "--output-dir", str(out)])


def test_run_nonexistent_target_errors(tmp_path):
    out = tmp_path / "out"
    with pytest.raises(SystemExit, match="not found"):
        main(["run", str(tmp_path / "ghost.py"), "--output-dir", str(out)])


def test_run_forwards_n_points_when_accepted(tmp_path):
    target = _write_target(
        tmp_path,
        "from pathlib import Path\n"
        "def main(output_dir, n_points=0):\n"
        "    Path(output_dir).joinpath('npts').write_text(str(n_points))\n",
    )
    out = tmp_path / "out"
    rc = main(["run", str(target), "--output-dir", str(out), "--n-points", "7"])
    assert rc == 0
    assert (out / "npts").read_text() == "7"


def test_run_skips_n_points_when_target_rejects_kwarg(tmp_path):
    target = _write_target(
        tmp_path,
        "from pathlib import Path\n"
        "def main(output_dir):\n"
        "    Path(output_dir).joinpath('ok').write_text('1')\n",
    )
    out = tmp_path / "out"
    rc = main(["run", str(target), "--output-dir", str(out), "--n-points", "9"])
    assert rc == 0
    assert (out / "ok").read_text() == "1"
