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


# ---------------------------------------------------------------------------
# `polygram batch` subcommand
# ---------------------------------------------------------------------------


def _build_separation_graph_for_animals_hea(tmp_path: Path) -> Path:
    from polygram.analysis import build_separation_graph, triage_dictionary

    from examples.animals_hea import build_dictionary

    d = build_dictionary()
    graph = build_separation_graph(
        triage_dictionary(d), threshold=0.0, include_within_cluster=True
    )
    p = tmp_path / "sep.json"
    p.write_text(graph.to_json())
    return p


class TestBatchSubcommand:
    def test_end_to_end(self, tmp_path: Path, capsys):
        graph_path = _build_separation_graph_for_animals_hea(tmp_path)
        out = tmp_path / "out"
        rc = main(
            [
                "batch",
                "--feature-graph",
                str(graph_path),
                "--dictionary",
                "examples.animals_hea:build_dictionary",
                "--top-k",
                "2",
                "--knobs",
                "cluster_shared",
                "--output-dir",
                str(out),
            ]
        )
        assert rc == 0
        results_path = out / "batch_results.json"
        assert results_path.is_file()
        data = json.loads(results_path.read_text())
        assert len(data["runs"]) == 2
        assert data["source_graph"]["kind"] == "separation"
        # Stdout names the resolved path.
        captured = capsys.readouterr()
        assert str(results_path) in captured.out

    def test_top_k_above_cap_rejected(self, tmp_path: Path):
        graph_path = _build_separation_graph_for_animals_hea(tmp_path)
        with pytest.raises(SystemExit):
            main(
                [
                    "batch",
                    "--feature-graph",
                    str(graph_path),
                    "--dictionary",
                    "examples.animals_hea:build_dictionary",
                    "--top-k",
                    "17",
                ]
            )

    def test_top_k_below_one_rejected(self, tmp_path: Path):
        graph_path = _build_separation_graph_for_animals_hea(tmp_path)
        with pytest.raises(SystemExit):
            main(
                [
                    "batch",
                    "--feature-graph",
                    str(graph_path),
                    "--dictionary",
                    "examples.animals_hea:build_dictionary",
                    "--top-k",
                    "0",
                ]
            )

    def test_unknown_knobs_rejected(self, tmp_path: Path):
        graph_path = _build_separation_graph_for_animals_hea(tmp_path)
        with pytest.raises(SystemExit):
            main(
                [
                    "batch",
                    "--feature-graph",
                    str(graph_path),
                    "--dictionary",
                    "examples.animals_hea:build_dictionary",
                    "--knobs",
                    "bogus",
                ]
            )

    def test_malformed_feature_graph_rejected(self, tmp_path: Path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not valid json")
        with pytest.raises(SystemExit, match="parse"):
            main(
                [
                    "batch",
                    "--feature-graph",
                    str(bad),
                    "--dictionary",
                    "examples.animals_hea:build_dictionary",
                ]
            )

    def test_dictionary_missing_graph_node_rejected(self, tmp_path: Path):
        # Hand-craft a graph that names a feature the dictionary
        # doesn't declare.
        from polygram.analysis.feature_graph import FeatureEdge, FeatureGraph

        graph = FeatureGraph(
            kind="separation",
            nodes=("dog_poodle", "ghost"),
            edges=(
                FeatureEdge(
                    source="dog_poodle",
                    target="ghost",
                    weight=0.6,
                    floor=0.5,
                    gap=0.1,
                    is_cross_cluster=True,
                    reason="irreducible_cross_cluster",
                ),
            ),
            clusters=(("dog_poodle", "ghost"),),
        )
        graph_path = tmp_path / "g.json"
        graph_path.write_text(graph.to_json())
        with pytest.raises(SystemExit, match="ghost"):
            main(
                [
                    "batch",
                    "--feature-graph",
                    str(graph_path),
                    "--dictionary",
                    "examples.animals_hea:build_dictionary",
                ]
            )

    def test_qorca_md_dictionary_path_rejected(self, tmp_path: Path):
        graph_path = _build_separation_graph_for_animals_hea(tmp_path)
        # An empty .q.orca.md file is enough to trigger the rejection
        # — the CLI catches the suffix before trying to parse.
        qpath = tmp_path / "fake.q.orca.md"
        qpath.write_text("# machine X\n")
        with pytest.raises(SystemExit, match="module:callable"):
            main(
                [
                    "batch",
                    "--feature-graph",
                    str(graph_path),
                    "--dictionary",
                    str(qpath),
                ]
            )


# ---------------------------------------------------------------------------
# `polygram sae-import` subcommand
# ---------------------------------------------------------------------------


def _synth_safetensors_for_cli(path: Path, *, n: int, d: int) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    rng = np.random.default_rng(42)
    save_file(
        {"W_dec": rng.standard_normal((n, d)).astype(np.float32)},
        str(path),
    )


class TestSaeImportSubcommand:
    def test_end_to_end_round_trips_through_load_toy_sae(
        self, tmp_path: Path
    ):
        from polygram import load_toy_sae

        src = tmp_path / "sae.safetensors"
        out = tmp_path / "picked.json"
        _synth_safetensors_for_cli(src, n=8, d=16)
        rc = main(
            [
                "sae-import",
                str(src),
                "--features",
                "0,1,4,5",
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        records = load_toy_sae(out)
        assert set(records.keys()) == {0, 1, 4, 5}
        assert all(records[k].projection.shape == (16,) for k in records)

    def test_chains_into_polygram_analyze(self, tmp_path: Path):
        # Spec scenario: `sae-import → analyze` works without further
        # plumbing.
        src = tmp_path / "sae.safetensors"
        picked = tmp_path / "picked.json"
        report = tmp_path / "report.md"
        _synth_safetensors_for_cli(src, n=8, d=16)
        rc = main(
            [
                "sae-import",
                str(src),
                "--features",
                "0,1,2,3",
                "--output",
                str(picked),
            ]
        )
        assert rc == 0
        rc = main(
            [
                "analyze",
                str(picked),
                "--features",
                "0,1,2,3",
                "--output",
                str(report),
            ]
        )
        assert rc == 0
        assert report.exists()

    def test_missing_source_rejected(self, tmp_path: Path):
        with pytest.raises(SystemExit, match="SAE file not found"):
            main(["sae-import", str(tmp_path / "ghost.safetensors")])

    def test_features_with_unknown_id_rejected(self, tmp_path: Path):
        src = tmp_path / "sae.safetensors"
        _synth_safetensors_for_cli(src, n=4, d=4)
        with pytest.raises(SystemExit, match=r"\[0, 4\)"):
            main(
                [
                    "sae-import",
                    str(src),
                    "--features",
                    "0,5",
                ]
            )

    def test_names_id_to_name_shape(self, tmp_path: Path):
        from polygram import load_toy_sae

        src = tmp_path / "sae.safetensors"
        out = tmp_path / "picked.json"
        labels = tmp_path / "labels.json"
        _synth_safetensors_for_cli(src, n=4, d=4)
        labels.write_text(json.dumps({"0": "dog_poodle", "2": "bird_hawk"}))
        rc = main(
            [
                "sae-import",
                str(src),
                "--names",
                str(labels),
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        recs = load_toy_sae(out)
        assert recs[0].name == "dog_poodle"
        assert recs[1].name == "feat_1"
        assert recs[2].name == "bird_hawk"

    def test_names_name_to_id_shape_inverted(self, tmp_path: Path):
        from polygram import load_toy_sae

        src = tmp_path / "sae.safetensors"
        out = tmp_path / "picked.json"
        labels = tmp_path / "labels.json"
        _synth_safetensors_for_cli(src, n=4, d=4)
        labels.write_text(
            json.dumps({"dog_poodle": 0, "bird_hawk": 2})
        )
        rc = main(
            [
                "sae-import",
                str(src),
                "--names",
                str(labels),
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        recs = load_toy_sae(out)
        assert recs[0].name == "dog_poodle"
        assert recs[2].name == "bird_hawk"

    def test_names_mixed_value_types_rejected(self, tmp_path: Path):
        src = tmp_path / "sae.safetensors"
        labels = tmp_path / "labels.json"
        _synth_safetensors_for_cli(src, n=4, d=4)
        labels.write_text(json.dumps({"0": "dog_poodle", "1": 5}))
        with pytest.raises(SystemExit, match="mixes value types"):
            main(["sae-import", str(src), "--names", str(labels)])

    def test_output_omitted_writes_to_stdout(
        self, tmp_path: Path, capsys
    ):
        src = tmp_path / "sae.safetensors"
        _synth_safetensors_for_cli(src, n=2, d=3)
        rc = main(["sae-import", str(src)])
        assert rc == 0
        out = capsys.readouterr().out
        # Output is JSON with the toy-SAE schema.
        data = json.loads(out)
        assert data["schema_version"] == 1
        assert len(data["features"]) == 2
