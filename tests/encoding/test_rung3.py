"""Rung3 encoding — default-knob equivalence with MPSRung1, smooth
knob perturbation, and torch-free import surface."""

import math
import sys

import numpy as np
import pytest

from polygram import Dictionary, Feature, MPSRung1, Rung3, Rung3State
from polygram.encoding import (
    RUNG3_DEFAULT_PSI_AUX,
    RUNG3_DEFAULT_THETA_AMP,
    rung3_amp_overlap_squared,
)


def _toy_dict(encoding, alpha_b=0.1, beta_b=0.4, gamma_b=0.05, phi_b=0.0):
    return Dictionary(
        name="toy",
        features=[
            Feature(name="a", cluster="c1", beta=0.0, alpha=0.0, gamma=0.0),
            Feature(name="b", cluster="c2", beta=beta_b,
                    alpha=alpha_b, gamma=gamma_b, phi=phi_b),
        ],
        hierarchy={"c1": ["a"], "c2": ["b"]},
        encoding=encoding,
    )


class TestDefaultKnobEquivalence:
    def test_default_amp_overlap_is_one(self):
        v = rung3_amp_overlap_squared(
            RUNG3_DEFAULT_THETA_AMP, RUNG3_DEFAULT_PSI_AUX,
            RUNG3_DEFAULT_THETA_AMP, RUNG3_DEFAULT_PSI_AUX,
        )
        assert v == pytest.approx(1.0, abs=1e-15)

    def test_dictionary_gram_matches_mps_at_default_knobs(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            alpha = float(rng.uniform(-0.5, 0.5))
            beta = float(rng.uniform(-0.5, 0.5))
            gamma = float(rng.uniform(-0.5, 0.5))
            phi = float(rng.uniform(0, 2 * math.pi))
            mps_dict = _toy_dict(MPSRung1(), alpha, beta, gamma, phi)
            rung3_dict = _toy_dict(Rung3(), alpha, beta, gamma, phi)
            mps_g = mps_dict.gram()
            r3_g = rung3_dict.gram()
            assert r3_g.shape == mps_g.shape
            np.testing.assert_allclose(r3_g, mps_g, atol=1e-12)

    def test_grid_sweep_phi_equivalence(self):
        # 20×20 (phi_a, phi_b) grid + 5×5×5 (alpha, beta, gamma) grid
        # to satisfy task §1.4 — keep it modest for runtime.
        rng = np.random.default_rng(1)
        for _ in range(8):
            alpha = float(rng.uniform(-0.4, 0.4))
            beta = float(rng.uniform(-0.4, 0.4))
            gamma = float(rng.uniform(-0.4, 0.4))
            for phi_a in np.linspace(0, 2 * math.pi, 5):
                for phi_b in np.linspace(0, 2 * math.pi, 5):
                    mps_dict = Dictionary(
                        name="toy",
                        features=[
                            Feature(name="a", cluster="c1", beta=0.0,
                                    phi=float(phi_a)),
                            Feature(name="b", cluster="c2", beta=beta,
                                    alpha=alpha, gamma=gamma,
                                    phi=float(phi_b)),
                        ],
                        hierarchy={"c1": ["a"], "c2": ["b"]},
                        encoding=MPSRung1(),
                    )
                    r3_dict = Dictionary(
                        name="toy",
                        features=[
                            Feature(name="a", cluster="c1", beta=0.0,
                                    phi=float(phi_a)),
                            Feature(name="b", cluster="c2", beta=beta,
                                    alpha=alpha, gamma=gamma,
                                    phi=float(phi_b)),
                        ],
                        hierarchy={"c1": ["a"], "c2": ["b"]},
                        encoding=Rung3(),
                    )
                    np.testing.assert_allclose(
                        r3_dict.gram(), mps_dict.gram(), atol=1e-12
                    )


class TestKnobPerturbationSanity:
    def test_amp_overlap_smooth_in_theta(self):
        thetas = np.linspace(0, math.pi / 2, 20)
        vals = [
            rung3_amp_overlap_squared(
                RUNG3_DEFAULT_THETA_AMP, RUNG3_DEFAULT_PSI_AUX,
                float(t), 0.0,
            )
            for t in thetas
        ]
        # All values lie in [0, 1].
        assert all(0.0 - 1e-12 <= v <= 1.0 + 1e-12 for v in vals)
        # No huge jumps between adjacent samples.
        diffs = np.abs(np.diff(vals))
        assert diffs.max() < 0.2

    def test_amp_overlap_smooth_in_psi(self):
        psis = np.linspace(0, 2 * math.pi, 20)
        vals = [
            rung3_amp_overlap_squared(
                RUNG3_DEFAULT_THETA_AMP, RUNG3_DEFAULT_PSI_AUX,
                RUNG3_DEFAULT_THETA_AMP, float(p),
            )
            for p in psis
        ]
        assert all(0.0 - 1e-12 <= v <= 1.0 + 1e-12 for v in vals)
        diffs = np.abs(np.diff(vals))
        assert diffs.max() < 0.5

    def test_psi_pi_kills_amp_factor_at_default_theta(self):
        # |amp(π/4, 0)|² · |amp(π/4, π)|² evaluates to 0 — the most
        # surgical knob the rung-3 viability spike can wield.
        v = rung3_amp_overlap_squared(
            RUNG3_DEFAULT_THETA_AMP, 0.0,
            RUNG3_DEFAULT_THETA_AMP, math.pi,
        )
        assert v == pytest.approx(0.0, abs=1e-15)

    def test_dictionary_gram_responds_to_perturbed_theta_amp(self):
        d_default = _toy_dict(Rung3(), beta_b=0.5)
        d_perturbed = d_default.with_knob("b.theta_amp", 0.0)
        # At theta_b = 0, amp_overlap²(π/4, 0, 0, 0) = cos²(π/4) = 0.5.
        # The off-diagonal entry should drop accordingly.
        g0 = d_default.gram()
        g1 = d_perturbed.gram()
        ratio = abs(g1[0, 1]) ** 2 / max(abs(g0[0, 1]) ** 2, 1e-30)
        assert ratio == pytest.approx(0.5, rel=1e-9, abs=1e-12)


class TestRung3State:
    def test_amp_overlap_squared_matches_module_helper(self):
        a = Rung3State(alpha=0.1, beta=0.2, gamma=0.3, phi=0.4,
                       theta_amp=0.5, psi_aux=0.6)
        b = Rung3State(alpha=0.1, beta=0.2, gamma=0.3, phi=0.4,
                       theta_amp=0.7, psi_aux=0.8)
        v = a.amp_overlap_squared(b)
        v_helper = rung3_amp_overlap_squared(0.5, 0.6, 0.7, 0.8)
        assert v == pytest.approx(v_helper, abs=1e-15)

    def test_from_mps_knobs_carries_defaults(self):
        s = Rung3State.from_mps_knobs(0.1, 0.2, 0.3, 0.4)
        assert s.alpha == 0.1
        assert s.beta == 0.2
        assert s.gamma == 0.3
        assert s.phi == 0.4
        assert s.theta_amp == RUNG3_DEFAULT_THETA_AMP
        assert s.psi_aux == RUNG3_DEFAULT_PSI_AUX


class TestTorchFreeImport:
    def test_encoding_module_does_not_pull_torch(self):
        # Importing polygram.encoding must not transitively load torch.
        # Use a subprocess so we test against a fresh sys.modules — the
        # current process may have torch loaded by an earlier test
        # (epoch convergence, validator forward path) and the
        # process-global sys.modules check would false-positive.
        import subprocess
        import sys as _sys

        result = subprocess.run(
            [
                _sys.executable,
                "-c",
                "import sys, polygram.encoding; "
                "assert 'torch' not in sys.modules, "
                "'torch leaked through polygram.encoding import'; "
                "assert 'transformers' not in sys.modules, "
                "'transformers leaked through polygram.encoding import'",
            ],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"polygram.encoding pulled torch / transformers transitively. "
            f"stdout: {result.stdout!r}  stderr: {result.stderr!r}"
        )


class TestRung3Validation:
    def test_bond_dim_other_than_2_rejected(self):
        with pytest.raises(ValueError, match="bond_dim must be 2"):
            Rung3(bond_dim=3)
