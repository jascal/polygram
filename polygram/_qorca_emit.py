"""Internal Q-Orca emit — render a Dictionary as a `.q.orca.md`-style
machine string and parse it into a `QMachineDef`.

The public `polygram.emit.write_qorca` (added in change
`experiment-interference-sweep`) layers the file-write + provenance
header on top of the renderer here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polygram.dictionary import Dictionary

_FEATURE_NAME_TO_SLUG = re.compile(r"[^A-Za-z0-9]+")


def feature_slug(name: str) -> str:
    """Lowercase, alnum+underscore slug used in event/state names."""
    s = _FEATURE_NAME_TO_SLUG.sub("_", name).strip("_").lower()
    if not s:
        raise ValueError(f"feature name {name!r} produces empty slug")
    return s


def render_machine_markdown(dictionary: Dictionary) -> str:
    """Render the Dictionary as a larql-animals-interference-style
    `.q.orca.md` string. No file I/O, no provenance header — that's the
    public emitter's job."""
    feats = dictionary.features
    slugs = [feature_slug(f.name) for f in feats]

    lines: list[str] = []
    lines.append(f"# machine {dictionary.name}")
    lines.append("")
    lines.append(
        f"Polygram-generated rung-1 MPS dictionary "
        f"({len(feats)} features, {len(dictionary.hierarchy)} clusters)."
    )
    lines.append("")

    lines.append("## context")
    lines.append("| Field    | Type        | Default            |")
    lines.append("|----------|-------------|--------------------|")
    lines.append("| qubits   | list<qubit> | [q0, q1, q2]       |")
    lines.append("")

    lines.append("## events")
    for slug in slugs:
        lines.append(f"- prepare_{slug}")
    lines.append("- measure_done")
    lines.append("")

    lines.append("## state idle [initial]")
    lines.append("> Concept register in `|000>`.")
    lines.append("")
    for f, slug in zip(feats, slugs):
        lines.append(f"## state prepared_{slug}")
        lines.append(
            f"> `|{f.name}>` prepared via "
            f"`prepare_concept({f.alpha}, {f.beta}, {f.gamma}, {f.phi})`. "
            f"Cluster: {f.cluster}."
        )
        lines.append("")
    lines.append("## state done [final]")
    lines.append("> Measurement collapsed the register.")
    lines.append("")

    lines.append("## transitions")
    lines.append("| Source | Event | Guard | Target | Action |")
    lines.append("|--------|-------|-------|--------|--------|")
    for f, slug in zip(feats, slugs):
        action_call = (
            f"prepare_concept({f.alpha}, {f.beta}, {f.gamma}, {f.phi})"
        )
        lines.append(
            f"| idle | prepare_{slug} | | prepared_{slug} | {action_call} |"
        )
    for slug in slugs:
        lines.append(f"| prepared_{slug} | measure_done | | done | |")
    lines.append("")

    lines.append("## actions")
    lines.append("| Name | Signature | Effect |")
    lines.append("|------|-----------|--------|")
    lines.append(
        "| prepare_concept "
        "| (qs, a: angle, b: angle, c: angle, phi: angle) -> qs "
        "| Ry(qs[0], a); CNOT(qs[0], qs[1]); Ry(qs[1], a + b); "
        "Rz(qs[1], phi); CNOT(qs[1], qs[2]); Ry(qs[2], b + c) |"
    )
    lines.append("")

    lines.append("## verification rules")
    lines.append("- unitarity")
    lines.append("- mps_bond_2_with_phase_knob")
    lines.append("")

    return "\n".join(lines)


def build_machine(dictionary: Dictionary):
    """Render + parse → returns a `q_orca.QMachineDef`."""
    from q_orca.parser.markdown_parser import parse_q_orca_markdown

    src = render_machine_markdown(dictionary)
    result = parse_q_orca_markdown(src)
    if result.errors:
        raise RuntimeError(
            "Polygram-generated machine failed to parse cleanly:\n"
            + "\n".join(result.errors)
            + "\n--- generated source ---\n"
            + src
        )
    return result.file.machines[0]
