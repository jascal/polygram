"""Public Q-Orca emitter — round-trips through q-orca parser + verifier."""

import math

from polygram.dictionary import Dictionary, Feature
from polygram.emit import write_qorca


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
