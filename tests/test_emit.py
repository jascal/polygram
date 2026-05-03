"""Public Q-Orca emitter — round-trips through q-orca parser + verifier."""

import math

from polygram.dictionary import Dictionary, Feature
from polygram.emit import write_qorca
from polygram.encoding import HEA_Rung2


def _animals() -> Dictionary:
    pi_2 = math.pi / 2
    return Dictionary(
        name="AnimalsEmitTest",
        features=[
            Feature("dog_at_rest", "dogs", beta=-0.5),
            Feature("dog_in_motion", "dogs", beta=-0.5, phi=pi_2),
            Feature("bird_at_rest", "birds", beta=0.5),
            Feature("bird_in_motion", "birds", beta=0.5, phi=pi_2),
        ],
        hierarchy={
            "dogs": ["dog_at_rest", "dog_in_motion"],
            "birds": ["bird_at_rest", "bird_in_motion"],
        },
    )


def test_write_qorca_creates_file(tmp_path):
    out = tmp_path / "animals.q.orca.md"
    p = write_qorca(_animals(), out)
    assert p == out
    assert out.exists()
    assert out.stat().st_size > 0


def test_emitted_file_contains_provenance_block(tmp_path):
    out = tmp_path / "animals.q.orca.md"
    write_qorca(_animals(), out)
    text = out.read_text()
    assert "Polygram-generated artifact" in text
    assert "source dictionary: AnimalsEmitTest" in text
    assert "git rev:" in text


def test_emitted_file_parses_clean(tmp_path):
    from q_orca.parser.markdown_parser import parse_q_orca_markdown

    out = tmp_path / "animals.q.orca.md"
    write_qorca(_animals(), out)
    result = parse_q_orca_markdown(out.read_text())
    assert not result.errors, result.errors
    assert len(result.file.machines) == 1
    machine = result.file.machines[0]
    assert machine.name == "AnimalsEmitTest"
    assert any(s.is_initial for s in machine.states)
    assert any(s.is_final for s in machine.states)
    assert len(machine.transitions) >= len(_animals().features) * 2


def test_emitted_file_uses_preparation_form(tmp_path):
    """When φ != 0 anywhere, transitions must be prep-form (`prepare_*`
    events into distinct `prepared_*` states), never inverse-form."""
    out = tmp_path / "animals.q.orca.md"
    write_qorca(_animals(), out)
    text = out.read_text()
    for slug in ("dog_at_rest", "dog_in_motion", "bird_at_rest", "bird_in_motion"):
        assert f"prepare_{slug}" in text
        assert f"prepared_{slug}" in text


def _hea_animals() -> Dictionary:
    return Dictionary(
        name="HeaEmitTest",
        features=[
            Feature("a", "s1", beta=0.10, alpha=0.05, gamma=0.02),
            Feature("b", "s1", beta=0.11, alpha=0.04, gamma=0.03),
            Feature("c", "s2", beta=1.20, alpha=1.10, gamma=1.00),
        ],
        hierarchy={"s1": ["a", "b"], "s2": ["c"]},
        encoding=HEA_Rung2(depth=2),
    )


def test_hea_round_trip_verifies_clean(tmp_path):
    from q_orca.parser.markdown_parser import parse_q_orca_markdown
    from q_orca.verifier import VerifyOptions, verify

    out = tmp_path / "hea.q.orca.md"
    write_qorca(_hea_animals(), out)
    parsed = parse_q_orca_markdown(out.read_text())
    assert not parsed.errors, parsed.errors

    machine = parsed.file.machines[0]
    assert machine.encoding.kind == "hea"
    assert [r.cluster for r in machine.theta.rows] == ["s1", "s1", "s2"]

    result = verify(machine, VerifyOptions(skip_resource_bounds=True))
    assert result.valid
    forbidden = {
        "HEA_GRAM_INVALID",
        "HEA_TIER_INVARIANT_VIOLATED",
        "HEA_TIER_UNDEFINED",
    }
    offenders = [e for e in result.errors if e.code in forbidden]
    assert not offenders, [(e.code, e.message) for e in offenders]
