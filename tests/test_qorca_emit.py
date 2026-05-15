"""HEA branch of `polygram._qorca_emit` — render + parse round-trip."""

from __future__ import annotations

import numpy as np

from polygram.dictionary import Dictionary, Feature
from polygram.encoding import HEA_Rung2
from polygram._qorca_emit import build_machine, render_machine_markdown


def _hea_dictionary(encoding: HEA_Rung2 | None = None) -> Dictionary:
    return Dictionary(
        name="HeaEmitFixture",
        features=[
            Feature("a", "s1", beta=0.10, alpha=0.05, gamma=0.02),
            Feature("b", "s1", beta=0.11, alpha=0.04, gamma=0.03),
            Feature("c", "s2", beta=1.20, alpha=1.10, gamma=1.00),
        ],
        hierarchy={"s1": ["a", "b"], "s2": ["c"]},
        encoding=encoding or HEA_Rung2(depth=2),
    )


class TestHEAEmit:
    def test_three_sections_emitted_in_order(self):
        text = render_machine_markdown(_hea_dictionary())
        i_encoding = text.index("## encoding")
        i_theta = text.index("## theta")
        i_invariants = text.index("## invariants")
        assert i_encoding < i_theta < i_invariants

    def test_encoding_table_has_kind_hea(self):
        text = render_machine_markdown(_hea_dictionary(HEA_Rung2(depth=3)))
        assert "| kind      | hea    |" in text
        assert "| depth     | 3" in text
        assert "| entangler | ring" in text
        assert "| rotations | Ry, Rz |" in text

    def test_theta_table_has_three_columns_with_cluster(self):
        text = render_machine_markdown(_hea_dictionary())
        assert "| concept | tensor | cluster |" in text
        # Each feature row carries the declared cluster verbatim.
        # Inspect the body lines in the theta block.
        body = text.split("## theta")[1].split("##")[0]
        rows = [
            line for line in body.splitlines()
            if line.startswith("| ") and "| concept" not in line and not line.startswith("|---")
        ]
        clusters = [line.rsplit("|", 2)[-2].strip() for line in rows]
        assert clusters == ["s1", "s1", "s2"]

    def test_parses_cleanly_into_qmachinedef(self):
        machine = build_machine(_hea_dictionary())
        assert machine.encoding is not None
        assert machine.encoding.kind == "hea"
        assert [r.cluster for r in machine.theta.rows] == ["s1", "s1", "s2"]
        assert len(machine.invariants) == 1
        inv = machine.invariants[0]
        assert inv.metric == "concept_gram_tier_separation"
        assert inv.op == "ge"
        assert inv.value == 0.025

    def test_explicit_theta_round_trips_through_emit_and_parse(self):
        encoding = HEA_Rung2(depth=2)
        custom = np.full(encoding.theta_shape, 0.123)
        d = Dictionary(
            name="HeaCustom",
            features=[
                Feature("a", "s1", beta=0.0, theta=custom),
                Feature("b", "s2", beta=0.0),
            ],
            hierarchy={"s1": ["a"], "s2": ["b"]},
            encoding=encoding,
        )
        machine = build_machine(d)
        parsed_a = np.asarray(machine.theta.rows[0].tensor)
        np.testing.assert_allclose(parsed_a, custom)

    def test_no_invariant(self):
        encoding = HEA_Rung2(depth=2, tier_separation_bound=None)
        text = render_machine_markdown(_hea_dictionary(encoding))
        assert "## invariants" not in text
        assert "concept_gram_tier_separation" not in text

    def test_one_query_concept_action_per_feature(self):
        text = render_machine_markdown(_hea_dictionary())
        transitions_block = text.split("## transitions")[1].split("##")[0]
        # One transition row per feature, each invoking `query_concept`.
        assert transitions_block.count("query_concept") == 3
        # Single action declaration in the actions table.
        assert text.count("| query_concept | (qs) -> qs") == 1

    def test_mps_branch_unchanged_for_default_encoding(self):
        d = Dictionary(
            name="StillRung1",
            features=[Feature("a", "s1", beta=0.0)],
            hierarchy={"s1": ["a"]},
        )
        text = render_machine_markdown(d)
        assert "## encoding" not in text
        assert "## theta" not in text
        assert "prepare_concept" in text


# ---------------------------------------------------------------------------
# Rung4 emit (add-rung4-encoding-mvp §6)
# ---------------------------------------------------------------------------


class TestRung4Emit:
    @staticmethod
    def _rung4_pair(*, theta_amp=(0.0, 0.0), psi_aux=(0.0, 0.0),
                    theta_amp_b=(0.0, 0.0), psi_amp_b=(0.0, 0.0)):
        from polygram import Dictionary, Feature
        from polygram.encoding import Rung4

        return Dictionary(
            name="Rung4EmitTest",
            features=[
                Feature("a", "ca", beta=-0.5, phi=0.3,
                        theta_amp=theta_amp[0], psi_aux=psi_aux[0],
                        theta_amp_b=theta_amp_b[0], psi_amp_b=psi_amp_b[0]),
                Feature("b", "cb", beta=0.5, phi=0.7,
                        theta_amp=theta_amp[1], psi_aux=psi_aux[1],
                        theta_amp_b=theta_amp_b[1], psi_amp_b=psi_amp_b[1]),
            ],
            hierarchy={"ca": ["a"], "cb": ["b"]},
            encoding=Rung4(),
        )

    def test_rung4_emits_mps_substrate_header(self):
        from polygram._qorca_emit import render_machine_markdown

        d = self._rung4_pair()
        md = render_machine_markdown(d)
        assert "rung-4 MPS-substrate" in md
        assert "amp branch" in md.lower()
        assert "prepare_concept" in md
        # The MPS staircase is present; amp branch is NOT emitted as
        # a separate action.
        assert "## encoding" not in md  # no HEA-style encoding tag
        assert "## theta" not in md     # no HEA-style theta tensor

    def test_rung4_emits_amp_branch_section_with_four_columns(self):
        from polygram._qorca_emit import render_machine_markdown

        d = self._rung4_pair(
            theta_amp=(0.11, 0.22),
            psi_aux=(0.33, 0.44),
            theta_amp_b=(0.55, 0.66),
            psi_amp_b=(0.77, 0.88),
        )
        md = render_machine_markdown(d)
        assert "## amp branch" in md
        block = md.rsplit("## amp branch", 1)[1]
        assert (
            "| concept | theta_amp | psi_aux | theta_amp_b | psi_amp_b |"
            in block
        )
        # Per-feature values present verbatim.
        assert "| a | 0.11 | 0.33 | 0.55 | 0.77 |" in block
        assert "| b | 0.22 | 0.44 | 0.66 | 0.88 |" in block

    def test_rung4_default_knobs_gram_matches_qorca_gram(self):
        """With default Rung4 amp knobs (all 0), the analytic Rung4
        gram equals the MPS gram, which is what q-orca produces from
        the emitted machine. Round-trip equality test."""
        import numpy as np

        from polygram._qorca_emit import build_machine

        try:
            from q_orca.compiler.concept_gram_mps import (
                compute_concept_gram_mps,
            )
        except ImportError:
            import pytest as _pytest

            _pytest.skip("q-orca compiler not available")

        d = self._rung4_pair()  # all amp knobs at default 0
        analytic_gram = d.gram()
        # Round-trip through q-orca's MPS gram path on the emitted machine.
        machine = build_machine(d)
        qorca_gram = compute_concept_gram_mps(
            machine, concept_action_label="prepare_concept"
        )
        # The analytic Rung4 gram with default amp knobs equals the MPS
        # gram of the same (α, β, γ, φ), which is what q-orca produces.
        np.testing.assert_allclose(analytic_gram, qorca_gram, atol=1e-10)

    def test_rung4_build_machine_tolerates_amp_branch_section(self):
        from polygram._qorca_emit import build_machine

        d = self._rung4_pair(
            theta_amp=(0.1, 0.2),
            psi_aux=(0.3, 0.4),
            theta_amp_b=(0.5, 0.6),
            psi_amp_b=(0.7, 0.8),
        )
        # Parses cleanly through q-orca (unknown `## amp branch`
        # section is silently ignored); raises if the parser
        # rejects the file.
        machine = build_machine(d)
        assert machine.name == "Rung4EmitTest"


# ---------------------------------------------------------------------------
# Rung3 emit (parallel to Rung4: substrate + `## amp branch` sidecar)
# ---------------------------------------------------------------------------


class TestRung3Emit:
    @staticmethod
    def _rung3_pair(*, theta_amp=(0.0, 0.0), psi_aux=(0.0, 0.0)):
        from polygram import Dictionary, Feature
        from polygram.encoding import Rung3

        return Dictionary(
            name="Rung3EmitTest",
            features=[
                Feature("a", "ca", beta=-0.5, phi=0.3,
                        theta_amp=theta_amp[0], psi_aux=psi_aux[0]),
                Feature("b", "cb", beta=0.5, phi=0.7,
                        theta_amp=theta_amp[1], psi_aux=psi_aux[1]),
            ],
            hierarchy={"ca": ["a"], "cb": ["b"]},
            encoding=Rung3(),
        )

    def test_rung3_emits_mps_substrate_header(self):
        from polygram._qorca_emit import render_machine_markdown

        md = render_machine_markdown(self._rung3_pair())
        assert "rung-3 MPS-substrate" in md
        assert "amp branch" in md.lower()
        assert "prepare_concept" in md
        assert "## encoding" not in md
        assert "## theta" not in md

    def test_rung3_emits_amp_branch_section_with_two_columns(self):
        from polygram._qorca_emit import render_machine_markdown

        md = render_machine_markdown(
            self._rung3_pair(theta_amp=(0.11, 0.22), psi_aux=(0.33, 0.44))
        )
        assert "## amp branch" in md
        block = md.rsplit("## amp branch", 1)[1]
        assert "| concept | theta_amp | psi_aux |" in block
        # Rung3 variant omits the Rung4-only columns.
        assert "theta_amp_b" not in block
        assert "psi_amp_b" not in block
        assert "| a | 0.11 | 0.33 |" in block
        assert "| b | 0.22 | 0.44 |" in block

    def test_rung3_build_machine_tolerates_amp_branch_section(self):
        from polygram._qorca_emit import build_machine

        d = self._rung3_pair(theta_amp=(0.1, 0.2), psi_aux=(0.3, 0.4))
        machine = build_machine(d)
        assert machine.name == "Rung3EmitTest"


# ---------------------------------------------------------------------------
# Rung1/HEA should NOT emit a `## amp branch` section.
# ---------------------------------------------------------------------------


def test_mps_rung1_emits_no_amp_branch_section():
    from polygram import Dictionary, Feature
    from polygram._qorca_emit import render_machine_markdown

    d = Dictionary(
        name="Rung1NoAmpBranch",
        features=[Feature("a", "s1", beta=0.0)],
        hierarchy={"s1": ["a"]},
    )
    assert "## amp branch" not in render_machine_markdown(d)


def test_hea_emits_no_amp_branch_section():
    from polygram._qorca_emit import render_machine_markdown

    assert "## amp branch" not in render_machine_markdown(_hea_dictionary())
