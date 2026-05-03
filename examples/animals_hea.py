"""Animals HEA example — Polygram emits a rung-2 HEA dictionary.

Same Animals shape as ``animals_interference.py``, but constructed with
``encoding=HEA_Rung2(depth=2)``. The default ``(α, β, γ, φ)`` knobs are
laid across the first HEA layer; siblings within a cluster pick up
small, near-identical rotations, and the two clusters are pulled apart
by a magnitude shift on β.

Writes ``AnimalsHea.q.orca.md`` and asserts the q-orca verifier accepts
the file (Stage 4b green, including the declared
``concept_gram_tier_separation >= 0.025`` invariant). Prints the
analytic tier separation for the dictionary.
"""

from __future__ import annotations

from pathlib import Path

from polygram import Dictionary, Feature, HEA_Rung2, write_qorca


def build_dictionary() -> Dictionary:
    return Dictionary(
        name="AnimalsHea",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.50, alpha=0.05, gamma=0.02),
            Feature("dog_beagle", "dogs", beta=-0.48, alpha=0.04, gamma=0.03),
            Feature("bird_hawk", "birds", beta=0.50, alpha=-0.04, gamma=0.02),
            Feature("bird_sparrow", "birds", beta=0.52, alpha=-0.03, gamma=0.01),
        ],
        hierarchy={
            "dogs": ["dog_poodle", "dog_beagle"],
            "birds": ["bird_hawk", "bird_sparrow"],
        },
        encoding=HEA_Rung2(depth=2),
    )


def main(output_dir: str | Path = "examples/output") -> None:
    from q_orca.parser.markdown_parser import parse_q_orca_markdown
    from q_orca.verifier import VerifyOptions, verify

    out_dir = Path(output_dir) / "animals_hea"
    out_dir.mkdir(parents=True, exist_ok=True)

    dictionary = build_dictionary()
    out_path = out_dir / "AnimalsHea.q.orca.md"
    write_qorca(dictionary, out_path)

    parsed = parse_q_orca_markdown(out_path.read_text())
    if parsed.errors:
        raise SystemExit(f"parse errors: {parsed.errors}")
    machine = parsed.file.machines[0]

    result = verify(machine, VerifyOptions(skip_resource_bounds=True))
    forbidden = {
        "HEA_GRAM_INVALID",
        "HEA_TIER_INVARIANT_VIOLATED",
        "HEA_TIER_UNDEFINED",
    }
    offenders = [e for e in result.errors if e.code in forbidden]
    assert result.valid, [(e.code, e.message) for e in result.errors]
    assert not offenders, [(e.code, e.message) for e in offenders]

    sep = dictionary.tier_separation()
    print(f"emitted: {out_path}")
    print(f"encoding: {dictionary.encoding}")
    print(f"tier_separation: {sep:.4f} (declared bound: "
          f"{dictionary.encoding.tier_separation_bound})")
    print("verify.valid: True")


if __name__ == "__main__":
    main()
