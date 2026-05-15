"""Tests for the add-phase-knob-assignment change.

The cornerstone falsifying invariant
(`test_phase_knobs_activate_mpsrung1_capacity`) is the load-bearing
assertion: with `encoding=MPSRung1()`, `from_sae_lens(...,
assign_phase_knobs=True)` MUST produce a gram measurably different
from `assign_phase_knobs=False` — Frobenius > 1.0 AND mean off-diagonal
drops to < 0.5× the default. Calibrated against the bug-repro sanity
check on the toy fixture: 0.76 → 0.28 (63% drop, 12 → 1 saturated pairs).

If the impl is a no-op or marginal, this test fails loudly.

Other tests cover: byte-identity at default; Rung3/Rung4 also-applicable;
HEA_Rung2 no-op; determinism; range bounds; degenerate-PCA fallback;
SAEImportConfig propagation; both-flags-on combinatorics.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from polygram import (
    HEA_Rung2,
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
# Cornerstone: phase-knob activation actually changes the gram
# ---------------------------------------------------------------------------


def test_phase_knobs_activate_mpsrung1_capacity():
    """The cornerstone test for add-phase-knob-assignment. This is the
    fix for the load-bearing finding from the 2026-05-15 GPT-2 bug
    report: MPSRung1.gram() saturates on activation-uncorrelated
    features because the loader leaves α and φ at 0.

    With `assign_phase_knobs=True`, the gram MUST differ from the
    default by Frobenius > 1.0 AND mean off-diagonal drops below
    half the default value (sanity check predicted 0.76 → 0.28 on
    the toy fixture)."""
    records, ids = _load_toy_records(8)

    d_off, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_phase_knobs=False,
    )
    d_on, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_phase_knobs=True,
    )

    g_off = np.abs(d_off.gram()) ** 2
    g_on = np.abs(d_on.gram()) ** 2

    full_fro = float(np.linalg.norm(g_off - g_on, ord="fro"))
    assert full_fro > 1.0, (
        f"MPSRung1 gram nearly identical under assign_phase_knobs=True "
        f"vs False — phase-knob assignment didn't engage. Frobenius: "
        f"{full_fro:.2e}"
    )

    n = g_off.shape[0]
    iu = np.triu_indices(n, k=1)
    mean_off = float(g_off[iu].mean())
    mean_on = float(g_on[iu].mean())
    assert mean_on < 0.5 * mean_off, (
        f"phase-knob assignment did not materially reduce gram "
        f"saturation: mean off-diag went {mean_off:.4f} → {mean_on:.4f} "
        f"(expected ≤ 0.5× per the bug-repro sanity check)"
    )


def test_phase_knobs_activate_rung3():
    """Same cornerstone shape for Rung3. Phase knobs apply to all
    MPS-substrate encodings."""
    records, ids = _load_toy_records(8)

    d_off, _ = from_sae_lens(
        records, ids, encoding=Rung3(), assign_phase_knobs=False,
    )
    d_on, _ = from_sae_lens(
        records, ids, encoding=Rung3(), assign_phase_knobs=True,
    )

    g_off = np.abs(d_off.gram()) ** 2
    g_on = np.abs(d_on.gram()) ** 2

    full_fro = float(np.linalg.norm(g_off - g_on, ord="fro"))
    assert full_fro > 1.0, f"Rung3 gram unchanged: Frobenius={full_fro:.2e}"


def test_phase_knobs_activate_rung4():
    """Same cornerstone shape for Rung4."""
    records, ids = _load_toy_records(8)

    d_off, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_phase_knobs=False,
    )
    d_on, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_phase_knobs=True,
    )

    g_off = np.abs(d_off.gram()) ** 2
    g_on = np.abs(d_on.gram()) ** 2

    full_fro = float(np.linalg.norm(g_off - g_on, ord="fro"))
    assert full_fro > 1.0, f"Rung4 gram unchanged: Frobenius={full_fro:.2e}"


# ---------------------------------------------------------------------------
# Byte-identity invariants
# ---------------------------------------------------------------------------


def test_default_false_is_byte_identical_to_pre_change():
    """`assign_phase_knobs=False` (the default) produces a Rung4 gram
    bit-identical to omitting the kwarg entirely."""
    records, ids = _load_toy_records(8)

    d_default, _ = from_sae_lens(records, ids, encoding=Rung4())
    d_explicit_false, _ = from_sae_lens(
        records, ids, encoding=Rung4(), assign_phase_knobs=False,
    )

    np.testing.assert_array_equal(d_default.gram(), d_explicit_false.gram())


def test_phase_knobs_true_is_no_op_for_hea_rung2():
    """`HEA_Rung2`'s per-feature θ tensor (rotations × depth × n_qubits)
    has a different shape than the MPS-substrate phase knobs. The flag
    is a structural no-op for HEA_Rung2."""
    records, ids = _load_toy_records(8)

    enc = HEA_Rung2(depth=1, n_qubits=3)
    d_off, _ = from_sae_lens(records, ids, encoding=enc, assign_phase_knobs=False)
    d_on, _ = from_sae_lens(records, ids, encoding=enc, assign_phase_knobs=True)

    np.testing.assert_array_equal(d_off.gram(), d_on.gram())


# ---------------------------------------------------------------------------
# Determinism + sanity
# ---------------------------------------------------------------------------


def test_phase_knob_assignment_is_deterministic():
    """Same input → same output."""
    records, ids = _load_toy_records(8)

    d_first, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_phase_knobs=True,
    )
    d_second, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_phase_knobs=True,
    )

    np.testing.assert_array_equal(d_first.gram(), d_second.gram())


def test_phase_knob_values_lie_in_per_knob_ranges():
    """α and φ both fall in `[0, 2π]`."""
    records, ids = _load_toy_records(8)

    d, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_phase_knobs=True,
    )

    for f in d.features:
        assert 0.0 <= f.alpha <= 2 * math.pi, (
            f"alpha out of [0, 2π]: {f.alpha}"
        )
        assert 0.0 <= f.phi <= 2 * math.pi, (
            f"phi out of [0, 2π]: {f.phi}"
        )


def test_phase_knob_values_vary_across_features():
    """Per-feature distinct α values — guards against an impl bug that
    writes the same value to every feature."""
    records, ids = _load_toy_records(8)

    d, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_phase_knobs=True,
    )

    alphas = [f.alpha for f in d.features]
    assert len(set(alphas)) > 1, (
        f"All alpha values are identical: {alphas[0]} — assignment isn't "
        f"producing per-feature variation"
    )


# ---------------------------------------------------------------------------
# Degenerate fallback
# ---------------------------------------------------------------------------


def test_degenerate_pca_falls_back_to_encoding_defaults():
    """A 2-feature SAE has at most 1 non-trivial PCA component. With
    `assign_phase_knobs=True` we want α and φ (which need PC2 and PC3
    respectively, both unavailable) to fall back to encoding defaults
    rather than crash."""
    records, ids = _load_toy_records(2)

    d, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_phase_knobs=True,
    )

    assert d is not None
    assert len(d.features) == 2
    gram = d.gram()
    assert gram.shape == (2, 2)
    # The diagonal of |gram|² is 1.0 (each state with itself);
    # off-diagonal is finite.
    gram_sq = np.abs(gram) ** 2
    assert np.allclose(np.diag(gram_sq), 1.0, atol=1e-9)
    assert np.isfinite(gram_sq[0, 1])


# ---------------------------------------------------------------------------
# SAEImportConfig propagation
# ---------------------------------------------------------------------------


def test_sae_import_config_propagates_assign_phase_knobs():
    """Passing `assign_phase_knobs` via `SAEImportConfig` produces the
    same Dictionary as passing it as a kwarg."""
    from polygram.config import SAEImportConfig

    records, ids = _load_toy_records(8)

    d_via_kwarg, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(), assign_phase_knobs=True,
    )
    d_via_config, _ = from_sae_lens(
        records, ids, encoding=MPSRung1(),
        config=SAEImportConfig(assign_phase_knobs=True),
    )

    np.testing.assert_array_equal(d_via_kwarg.gram(), d_via_config.gram())


# ---------------------------------------------------------------------------
# Both-flags-on combinatorics (task 9.4a — "FULL on Rung4")
# ---------------------------------------------------------------------------


def test_both_flags_on_rung4_populates_all_six_knob_channels():
    """The two flags compose additively. With both on for Rung4, every
    feature carries non-default values across all six MPS-substrate +
    amp-branch knob channels: alpha, phi, theta_amp, psi_aux,
    theta_amp_b, psi_amp_b.

    "Non-default" means at least one feature differs from the
    encoding's default for that knob (defaults: alpha=0, phi=0,
    theta_amp=π/4, psi_aux=0, theta_amp_b=π/4, psi_amp_b=0)."""
    records, ids = _load_toy_records(8)

    d, _ = from_sae_lens(
        records, ids, encoding=Rung4(),
        assign_phase_knobs=True,
        assign_amp_knobs=True,
    )

    defaults = {
        "alpha": 0.0,
        "phi": 0.0,
        "theta_amp": math.pi / 4,
        "psi_aux": 0.0,
        "theta_amp_b": math.pi / 4,
        "psi_amp_b": 0.0,
    }
    for knob, default_value in defaults.items():
        values = [getattr(f, knob) for f in d.features]
        # At least one feature differs from the default by more than
        # FP noise.
        max_dev = max(abs(v - default_value) for v in values)
        assert max_dev > 1e-6, (
            f"All features have {knob} ≈ default ({default_value}) under "
            f"both-flags-on Rung4 — that knob channel didn't get populated"
        )


def test_both_flags_on_rung4_distinct_from_amp_only():
    """Both flags on must produce a different gram than amp-only-on.
    Catches an impl bug where assign_phase_knobs is silently dropped
    on the combined path."""
    records, ids = _load_toy_records(8)

    d_amp_only, _ = from_sae_lens(
        records, ids, encoding=Rung4(),
        assign_phase_knobs=False, assign_amp_knobs=True,
    )
    d_both, _ = from_sae_lens(
        records, ids, encoding=Rung4(),
        assign_phase_knobs=True, assign_amp_knobs=True,
    )

    g_amp = np.abs(d_amp_only.gram()) ** 2
    g_both = np.abs(d_both.gram()) ** 2
    fro = float(np.linalg.norm(g_amp - g_both, ord="fro"))
    assert fro > 0.1, (
        f"both-flags-on Rung4 gram ≈ amp-only-on gram (Frobenius={fro:.2e}); "
        f"phase-knob component dropped on the combined path"
    )
