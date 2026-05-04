"""Tests for the `polygram` console script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from polygram.cli import main

FIXTURE = Path("tests/fixtures/toy_sae.json")


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


def test_analyze_emits_sharing_graph(tmp_path: Path):
    report = tmp_path / "report.md"
    sharing = tmp_path / "sharing.json"
    rc = main(
        [
            "analyze",
            str(FIXTURE),
            "--features",
            "0,1,4,5",
            "--output",
            str(report),
            "--sharing-graph",
            str(sharing),
            "--sharing-threshold",
            "0.4",
        ]
    )
    assert rc == 0
    assert report.exists()
    data = json.loads(sharing.read_text())
    assert data["kind"] == "sharing"
    for key in ("kind", "nodes", "edges", "clusters", "metadata"):
        assert key in data
    assert data["metadata"]["threshold"] == 0.4


def test_analyze_emits_separation_graph(tmp_path: Path):
    report = tmp_path / "report.md"
    sep = tmp_path / "sep.json"
    rc = main(
        [
            "analyze",
            str(FIXTURE),
            "--features",
            "0,1,4,5",
            "--output",
            str(report),
            "--separation-graph",
            str(sep),
            "--separation-threshold",
            "0.15",
        ]
    )
    assert rc == 0
    data = json.loads(sep.read_text())
    assert data["kind"] == "separation"
    assert data["metadata"]["threshold"] == 0.15


def test_analyze_emits_both_graphs(tmp_path: Path):
    report = tmp_path / "report.md"
    sharing = tmp_path / "sharing.json"
    sep = tmp_path / "sep.json"
    rc = main(
        [
            "analyze",
            str(FIXTURE),
            "--features",
            "0,1,4,5",
            "--output",
            str(report),
            "--sharing-graph",
            str(sharing),
            "--separation-graph",
            str(sep),
        ]
    )
    assert rc == 0
    sharing_data = json.loads(sharing.read_text())
    sep_data = json.loads(sep.read_text())
    assert sharing_data["kind"] == "sharing"
    assert sep_data["kind"] == "separation"


def test_analyze_threshold_malformed(tmp_path: Path):
    report = tmp_path / "report.md"
    sharing = tmp_path / "sharing.json"
    with pytest.raises(SystemExit):
        main(
            [
                "analyze",
                str(FIXTURE),
                "--features",
                "0,1,4,5",
                "--output",
                str(report),
                "--sharing-graph",
                str(sharing),
                "--sharing-threshold",
                "not-a-float",
            ]
        )
