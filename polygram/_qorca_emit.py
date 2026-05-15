"""Internal Q-Orca emit — render a Dictionary as a `.q.orca.md`-style
machine string and parse it into a `QMachineDef`.

The public `polygram.emit.write_qorca` (added in change
`experiment-interference-sweep`) layers the file-write + provenance
header on top of the renderer here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from polygram.encoding import HEA_Rung2

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
    """Render the Dictionary as a `.q.orca.md` string.

    Dispatches on ``dictionary.encoding``: ``MPSRung1`` produces the
    larql-animals-interference-style staircase machine; ``HEA_Rung2``
    produces the larql-hea-minimal-style ``## encoding`` + ``## theta``
    machine. No file I/O, no provenance header — that's the public
    emitter's job.
    """
    if isinstance(dictionary.encoding, HEA_Rung2):
        return _render_hea_markdown(dictionary)
    return _render_mps_rung1_markdown(dictionary)


def _render_mps_rung1_markdown(dictionary: Dictionary) -> str:
    from polygram.encoding import Rung3, Rung4

    feats = dictionary.features
    slugs = [feature_slug(f.name) for f in feats]

    # Encoding label for the header. Rung3 and Rung4 both fall through
    # this MPS renderer for the (α, β, γ, φ) substrate; the amp branch
    # is captured in the trailing `## amp branch` section below so the
    # file round-trips through polygram without information loss.
    # q-orca's parser tolerates the unknown section and ignores it,
    # which is why `Dictionary.gram()` continues to apply the amp
    # factor analytically rather than via the q-orca compile path.
    if isinstance(dictionary.encoding, Rung4):
        encoding_label = "rung-4 MPS-substrate"
        amp_note = (
            " The rung-4 product-amp branch on q3/q4 is captured in "
            "the `## amp branch` section below; q-orca's gram path "
            "ignores that section and returns the MPSRung1-equivalent "
            "gram on (α, β, γ, φ). Polygram's `Dictionary.gram()` "
            "applies the product-amp factor on top per "
            "`polygram.encoding.rung4_amp_overlap`."
        )
    elif isinstance(dictionary.encoding, Rung3):
        encoding_label = "rung-3 MPS-substrate"
        amp_note = (
            " The rung-3 Bell-pattern amp branch on q3/q4 is captured "
            "in the `## amp branch` section below; q-orca's gram path "
            "ignores that section and returns the MPSRung1-equivalent "
            "gram on (α, β, γ, φ). Polygram's `Dictionary.gram()` "
            "applies the amp factor on top per "
            "`polygram.encoding.rung3_amp_overlap`."
        )
    else:
        encoding_label = "rung-1 MPS"
        amp_note = ""

    lines: list[str] = []
    lines.append(f"# machine {dictionary.name}")
    lines.append("")
    lines.append(
        f"Polygram-generated {encoding_label} dictionary "
        f"({len(feats)} features, {len(dictionary.hierarchy)} clusters)."
        f"{amp_note}"
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

    if isinstance(dictionary.encoding, (Rung3, Rung4)):
        lines.append("## amp branch")
        if isinstance(dictionary.encoding, Rung4):
            lines.append(
                "| concept | theta_amp | psi_aux | theta_amp_b | psi_amp_b |"
            )
            lines.append(
                "|---------|-----------|---------|-------------|-----------|"
            )
            for f, slug in zip(feats, slugs):
                lines.append(
                    f"| {slug} | {f.theta_amp} | {f.psi_aux} "
                    f"| {f.theta_amp_b} | {f.psi_amp_b} |"
                )
        else:
            lines.append("| concept | theta_amp | psi_aux |")
            lines.append("|---------|-----------|---------|")
            for f, slug in zip(feats, slugs):
                lines.append(f"| {slug} | {f.theta_amp} | {f.psi_aux} |")
        lines.append("")

    return "\n".join(lines)


def _render_hea_markdown(dictionary: Dictionary) -> str:
    from polygram.dictionary import _default_hea_theta

    encoding = dictionary.encoding
    assert isinstance(encoding, HEA_Rung2)
    feats = dictionary.features
    slugs = [feature_slug(f.name) for f in feats]
    n_qubits = encoding.n_qubits

    lines: list[str] = []
    lines.append(f"# machine {dictionary.name}")
    lines.append("")
    lines.append(
        f"Polygram-generated rung-2 HEA dictionary "
        f"({len(feats)} features, {len(dictionary.hierarchy)} clusters; "
        f"depth={encoding.depth}, entangler={encoding.entangler}, "
        f"rotations={list(encoding.rotations)})."
    )
    lines.append("")

    qubit_list = "[" + ", ".join(f"q{i}" for i in range(n_qubits)) + "]"
    qubit_pad = " " * max(0, len(qubit_list) - len("[q0, q1, q2]"))
    lines.append("## context")
    lines.append("| Field  | Type        | Default          |")
    lines.append("|--------|-------------|------------------|")
    lines.append(f"| qubits | list<qubit> | {qubit_list}{qubit_pad} |")
    lines.append("")

    lines.append("## events")
    for slug in slugs:
        lines.append(f"- prep_{slug}")
    lines.append("")

    lines.append("## state idle [initial]")
    lines.append(
        f"> Ground state of the {n_qubits}-qubit register, "
        f"before any concept preparation."
    )
    lines.append("")
    for f, slug in zip(feats, slugs):
        lines.append(f"## state queried_{slug} [final]")
        lines.append(
            f"> Register holds concept `{f.name}` (cluster: {f.cluster})."
        )
        lines.append("")

    lines.append("## transitions")
    lines.append("| Source | Event   | Guard | Target       | Action        |")
    lines.append("|--------|---------|-------|--------------|---------------|")
    for slug in slugs:
        lines.append(
            f"| idle   | prep_{slug} |       | queried_{slug} | query_concept |"
        )
    lines.append("")

    lines.append("## actions")
    lines.append("| Name          | Signature   |")
    lines.append("|---------------|-------------|")
    lines.append("| query_concept | (qs) -> qs  |")
    lines.append("")

    lines.append("## encoding")
    lines.append("| key       | value  |")
    lines.append("|-----------|--------|")
    lines.append("| kind      | hea    |")
    lines.append(f"| depth     | {encoding.depth}      |")
    lines.append(f"| entangler | {encoding.entangler}   |")
    rotations_str = ", ".join(encoding.rotations)
    lines.append(f"| rotations | {rotations_str} |")
    lines.append("")

    lines.append("## theta")
    lines.append("| concept | tensor | cluster |")
    lines.append("|---------|--------|---------|")
    for f, slug in zip(feats, slugs):
        theta = f.theta if f.theta is not None else _default_hea_theta(f, encoding)
        tensor_repr = _theta_to_literal(theta)
        lines.append(f"| {slug} | {tensor_repr} | {f.cluster} |")
    lines.append("")

    if encoding.tier_separation_bound is not None:
        lines.append("## invariants")
        lines.append(
            f"- concept_gram_tier_separation >= {encoding.tier_separation_bound}"
        )
        lines.append("")

    return "\n".join(lines)


def _theta_to_literal(theta) -> str:
    """Render a θ tensor as a literal-eval-able Python list-of-lists string."""
    return repr(theta.tolist())


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
