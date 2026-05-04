"""Pure-classical triage layer (`polygram.analysis`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from polygram import Cancellation, load_toy_sae
from polygram.analysis import (
    KNOB_SELECTION_GUIDANCE,
    SUITABILITY_FORMULA,
    encoding_suitability_score,
    feature_sensitivity,
    predict_cancellation_depth,
    render_report,
)
from polygram.cli import main as cli_main

FIXTURE = Path("tests/fixtures/toy_sae.json")


@pytest.fixture
def records():
    return load_toy_sae(FIXTURE)


def test_predicted_floor_matches_cancellation_for_target_pair(records):
    feature_ids = [0, 1, 4, 5]  # 2 mammals + 2 birds in toy fixture
    prediction = predict_cancellation_depth(records, feature_ids)

    a_name = prediction.dictionary.features[0].name
    b_name = prediction.dictionary.features[2].name  # cross-cluster pair
    pair = next(
        p for p in prediction.pairs
        if {p.feature_a, p.feature_b} == {a_name, b_name}
    )

    canc = Cancellation(
        dictionary=prediction.dictionary,
        target_pair=(a_name, b_name),
    )
    assert abs(canc.structural_floor() - pair.structural_floor) < 1e-9


def test_per_feature_sensitivity_is_mean_abs_v_over_pairs(records):
    feature_ids = [0, 1, 4, 5]
    prediction = predict_cancellation_depth(records, feature_ids)

    name = prediction.dictionary.features[0].name
    pairs_with = [
        p for p in prediction.pairs
        if name in (p.feature_a, p.feature_b)
    ]
    expected = float(np.mean([abs(p.V) for p in pairs_with]))
    assert abs(prediction.feature_sensitivity[name] - expected) < 1e-12


def test_encoding_suitability_score_in_unit_interval(records):
    score = encoding_suitability_score(records, [0, 1, 4, 5])
    assert 0.0 <= score <= 1.0


def test_pair_decomposition_obeys_M_plus_V_identity(records):
    prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
    for p in prediction.pairs:
        assert abs(p.M + p.V - p.current_overlap) < 1e-9
        assert abs(p.M - p.V - p.m_pi) < 1e-9
        assert abs(p.structural_floor - (p.M - abs(p.V))) < 1e-9
        assert p.cancellation_gap >= -1e-12


def test_feature_sensitivity_helper_matches_full_prediction(records):
    full = predict_cancellation_depth(records, [0, 1, 4, 5])
    helper = feature_sensitivity(records, [0, 1, 4, 5])
    assert helper == full.feature_sensitivity


def test_render_report_contains_required_sections(records):
    prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
    text = render_report(
        prediction, sae_path=str(FIXTURE), feature_ids=[0, 1, 4, 5]
    )
    assert "# Polygram analysis" in text
    assert "## Caveats" in text
    assert "## Pair predictions" in text
    assert "## Per-feature sensitivity" in text
    assert "## Choosing knobs" in text
    assert "## Encoding suitability" in text
    assert "rung-1" in text  # caveats mention the assumption
    assert SUITABILITY_FORMULA.splitlines()[0] in text


def test_render_report_quotes_knob_selection_guidance(records):
    prediction = predict_cancellation_depth(records, [0, 1, 4, 5])
    text = render_report(
        prediction, sae_path=str(FIXTURE), feature_ids=[0, 1, 4, 5]
    )
    assert KNOB_SELECTION_GUIDANCE in text
    assert "cluster-shatterer" in text


def test_knob_selection_guidance_constant_exposed():
    assert isinstance(KNOB_SELECTION_GUIDANCE, str)
    assert KNOB_SELECTION_GUIDANCE  # non-empty
    for required in (
        "cluster-shatterer",
        "Structural floor",
        "Rz",
        "<cluster>.phi",
    ):
        assert required in KNOB_SELECTION_GUIDANCE, (
            f"missing required substring: {required!r}"
        )


def test_predict_refuses_oversized_subset(records):
    with pytest.raises(ValueError, match="caps a Dictionary"):
        predict_cancellation_depth(records, list(range(9)))


def test_cli_analyze_writes_report(tmp_path: Path):
    out = tmp_path / "report.md"
    rc = cli_main(
        [
            "analyze",
            str(FIXTURE),
            "--features",
            "0,1,4,5",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text()
    assert "# Polygram analysis" in text
    assert "## Pair predictions" in text


def test_cli_analyze_rejects_bad_features(tmp_path: Path):
    out = tmp_path / "report.md"
    with pytest.raises(SystemExit, match="comma-separated ints"):
        cli_main(
            [
                "analyze",
                str(FIXTURE),
                "--features",
                "1,2,abc",
                "--output",
                str(out),
            ]
        )


def test_cli_analyze_missing_path(tmp_path: Path):
    with pytest.raises(SystemExit, match="SAE file not found"):
        cli_main(
            [
                "analyze",
                str(tmp_path / "ghost.json"),
                "--features",
                "0,1",
                "--output",
                str(tmp_path / "r.md"),
            ]
        )
