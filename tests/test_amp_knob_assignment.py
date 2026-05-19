"""Tests for the encoding-aware-knob-assignment change.

Load-bearing assertion: when `from_sae_lens(...,
assign_amp_knobs=True)` is called on a higher-rung encoding
(Rung3 or Rung4), the resulting `Dictionary.gram()` is
measurably different from the same call with
`assign_amp_knobs=False`. Without this, the higher rungs alias
MPSRung1 — the bug that `docs/research/rung4-viability-spike-v2.md`
documented.

The falsifying invariant (Frobenius distance > 1e-3 on both the
full matrix and the off-diagonal only) is the cornerstone test.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from polygram import (
    MPSRung1,
    Rung3,
    Rung4,
    from_sae_lens,
    load_toy_sae,
)


FIXTURE_TOY = Path(__file__).parent / "fixtures" / "toy_sae.json"


def _load_toy_records(n: int):
    records = load_toy_sae(FIXTURE_TOY)
    ids = sorted(records.keys())[:n]
    return records, ids


# ---------------------------------------------------------------------------
# Cornerstone: amp-knob activation actually changes the gram
# ---------------------------------------------------------------------------


def test_amp_knobs_activate_higher_rung_capacity_rung4():
    """The cornerstone test for encoding-aware-knob-assignment.

    A Rung4 dictionary built with `assign_amp_knobs=True` MUST
    produce a gram measurably different from the same dictionary
    built with `assign_amp_knobs=False`. Pre-this-change, the two
    were bit-identical because the loader left amp-branch knobs at
    their MPS-collapsing defaults — that bug is what
    `docs/research/rung4-viability-spike-v2.md` flagged."""
    records, ids = _load_toy_records(8)

    d_off, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=False,
    )
    d_on, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )

    g_off = np.abs(d_off.gram()) ** 2
    g_on = np.abs(d_on.gram()) ** 2

    # Full-matrix Frobenius distance well above FP noise.
    full_fro = float(np.linalg.norm(g_off - g_on, ord="fro"))
    assert full_fro > 1e-3, (
        f"Rung4 gram bit-identical under assign_amp_knobs=True vs False — "
        f"the amp-knob assignment didn't engage. Frobenius distance: "
        f"{full_fro:.2e}"
    )

    # Off-diagonal-only Frobenius (per the proposal's stronger
    # assertion — guards against a degenerate impl that perturbed
    # only on-diagonal terms).
    n = g_off.shape[0]
    iu = np.triu_indices(n, k=1)
    off_fro = float(np.linalg.norm(g_off[iu] - g_on[iu]))
    assert off_fro > 1e-3, (
        f"Rung4 amp-knob change concentrated on the diagonal — should "
        f"reach off-diagonal pairs. Off-diagonal Frobenius: {off_fro:.2e}"
    )


def test_amp_knobs_activate_higher_rung_capacity_rung3():
    """Same as the cornerstone test, for Rung3 (which has only branch-A
    amp knobs — theta_amp + psi_aux). Frobenius distance must still
    be > 1e-3."""
    records, ids = _load_toy_records(8)

    d_off, _ = from_sae_lens(
        records, ids, encoding=Rung3(), assign_amp_knobs=False,
    )
    d_on, _ = from_sae_lens(
        records, ids, encoding=Rung3(), assign_amp_knobs=True,
    )

    g_off = np.abs(d_off.gram()) ** 2
    g_on = np.abs(d_on.gram()) ** 2

    full_fro = float(np.linalg.norm(g_off - g_on, ord="fro"))
    assert full_fro > 1e-3, (
        f"Rung3 gram bit-identical under assign_amp_knobs=True vs False. "
        f"Frobenius distance: {full_fro:.2e}"
    )

    n = g_off.shape[0]
    iu = np.triu_indices(n, k=1)
    off_fro = float(np.linalg.norm(g_off[iu] - g_on[iu]))
    assert off_fro > 1e-3, (
        f"Rung3 amp-knob change concentrated on the diagonal. "
        f"Off-diagonal Frobenius: {off_fro:.2e}"
    )


# ---------------------------------------------------------------------------
# Byte-identity invariants
# ---------------------------------------------------------------------------


def test_default_matches_explicit_true_post_sm_sae():
    """`assign_amp_knobs=True` is the default since the sm-sae
    Recommended-defaults change. Omitting the kwarg is byte-identical
    to passing `True`; this test pins that invariant."""
    records, ids = _load_toy_records(8)

    d_default, _ = from_sae_lens(records, ids, encoding=Rung4())
    d_explicit_true, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )

    g_default = d_default.gram()
    g_explicit_true = d_explicit_true.gram()

    np.testing.assert_array_equal(g_default, g_explicit_true)


def test_amp_knobs_true_is_no_op_for_mpsrung1():
    """`MPSRung1` has no amp branch; `assign_amp_knobs=True` is a
    structural no-op. Gram is unchanged."""
    records, ids = _load_toy_records(8)

    d_off, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_amp_knobs=False,
    )
    d_on, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_amp_knobs=True,
    )

    np.testing.assert_array_equal(d_off.gram(), d_on.gram())


# ---------------------------------------------------------------------------
# Determinism + sanity
# ---------------------------------------------------------------------------


def test_amp_knob_assignment_is_deterministic():
    """Same input → same output across multiple calls."""
    records, ids = _load_toy_records(8)

    d_first, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )
    d_second, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )

    np.testing.assert_array_equal(d_first.gram(), d_second.gram())


def test_amp_knob_values_lie_in_per_knob_ranges():
    """Each assigned amp-branch knob value falls in its natural
    range: `theta_amp ∈ [0, π/2]`, `psi_aux ∈ [0, 2π]`, etc."""
    records, ids = _load_toy_records(8)

    d, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )

    for f in d.features:
        assert 0.0 <= f.theta_amp <= math.pi / 2, (
            f"theta_amp out of [0, π/2]: {f.theta_amp}"
        )
        assert 0.0 <= f.psi_aux <= 2 * math.pi, (
            f"psi_aux out of [0, 2π]: {f.psi_aux}"
        )
        assert 0.0 <= f.theta_amp_b <= math.pi / 2, (
            f"theta_amp_b out of [0, π/2]: {f.theta_amp_b}"
        )
        assert 0.0 <= f.psi_amp_b <= 2 * math.pi, (
            f"psi_amp_b out of [0, 2π]: {f.psi_amp_b}"
        )


def test_amp_knob_values_vary_across_features():
    """A working assignment produces distinct per-feature amp-knob
    values — not all identical. (An impl bug that wrote the same
    value to every feature would pass the byte-identity-against-
    MPSRung1 test but fail this one.)"""
    records, ids = _load_toy_records(8)

    d, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )

    theta_amps = [f.theta_amp for f in d.features]
    assert len(set(theta_amps)) > 1, (
        f"All theta_amp values are identical: {theta_amps[0]} — "
        f"the assignment isn't producing per-feature variation"
    )


# ---------------------------------------------------------------------------
# Degenerate fallback
# ---------------------------------------------------------------------------


def test_degenerate_pca_falls_back_to_encoding_defaults():
    """A 2-feature SAE has at most 1 non-trivial PCA axis (after
    centering). With encoding=Rung4 (which wants 4 amp-knob axes
    starting from axis-2), the call should not raise — the
    unavailable axes fall back to encoding defaults."""
    records, ids = _load_toy_records(2)

    d, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )

    # No crash, valid Dictionary. The fallback knob values are the
    # encoding defaults (π/4 for theta_amp, 0 for psi_aux, etc.) —
    # we don't assert specific values here, just that the call
    # completed and produced a usable Dictionary.
    assert d is not None
    assert len(d.features) == 2
    gram = d.gram()
    assert gram.shape == (2, 2)
    # Diagonal of |gram|² should be 1.0 (each state with itself);
    # off-diagonal should be a finite number.
    gram_sq = np.abs(gram) ** 2
    assert np.allclose(np.diag(gram_sq), 1.0, atol=1e-9)
    assert np.isfinite(gram_sq[0, 1])


# ---------------------------------------------------------------------------
# SAEImportConfig propagation
# ---------------------------------------------------------------------------


def test_sae_import_config_propagates_assign_amp_knobs():
    """Passing `assign_amp_knobs` via `SAEImportConfig` produces
    the same Dictionary as passing it via the kwarg."""
    from polygram.config import SAEImportConfig

    records, ids = _load_toy_records(8)

    d_via_kwarg, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )
    d_via_config, _ = from_sae_lens(
        records, ids, encoding=Rung4(),
        config=SAEImportConfig(assign_amp_knobs=True),
    )

    np.testing.assert_array_equal(d_via_kwarg.gram(), d_via_config.gram())


# ---------------------------------------------------------------------------
# Cross-encoding sanity
# ---------------------------------------------------------------------------


def test_rung4_amp_on_distinct_from_rung3_amp_on():
    """Rung4 with amp_knobs=True populates 4 amp knobs (branch-A +
    branch-B); Rung3 populates only branch-A. The grams should be
    distinct even though both use the same projection geometry."""
    records, ids = _load_toy_records(8)

    d_r3, _ = from_sae_lens(
        records, ids, encoding=Rung3(), assign_amp_knobs=True,
    )
    d_r4, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_amp_knobs=True,
    )

    g_r3 = np.abs(d_r3.gram()) ** 2
    g_r4 = np.abs(d_r4.gram()) ** 2

    full_fro = float(np.linalg.norm(g_r3 - g_r4, ord="fro"))
    # Distinct dictionaries; the threshold here is conservative.
    assert full_fro > 1e-4, (
        f"Rung3 and Rung4 amp-on dictionaries produce nearly identical "
        f"gram (Frobenius={full_fro:.2e}); branch-B knobs aren't "
        f"contributing"
    )
