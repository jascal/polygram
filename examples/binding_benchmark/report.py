"""Render a complete three-seed research report from saved empirical results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def label(name):
    if name == "identity":
        return "Host / identity"
    if name.startswith("compressed"):
        return "Polygram compressed"
    if name.startswith("sae"):
        return "SAE (128)"
    if name.startswith("tpr"):
        return "Supervised TPR (64)"
    return name.split("-seed")[0]


def render(root):
    summaries = [json.loads((root / f"eval-seed{s}/summary.json").read_text()) for s in range(3)]
    forges = [json.loads((root / f"forge-target32-seed{s}/report.json").read_text()) for s in range(3)]
    capture = json.loads((root / "capture-test/capture.json").read_text())
    sparsity = json.loads((root / "sae-training-diagnostics.json").read_text())
    amendments = json.loads((root / "protocol-amendments.json").read_text())
    host_amendment = next(item for item in amendments if item["stage"].startswith("host selection"))
    protocol = capture["protocol"]
    methods = ["identity", "sae", "compressed", "tpr", "pca32", "pca64", "pca_matched",
               "learned32", "learned64", "learned_matched", "pca128", "learned128"]

    def get(seed, method):
        summary = summaries[seed]
        if method == "identity":
            return summary["identity"]
        if method in ("sae", "tpr"):
            return summary[f"{method}-seed{seed}"]
        if method == "compressed":
            return summary[f"compressed-target32-seed{seed}"]
        if method.endswith("_matched"):
            rank = get(seed, "compressed")["capacity"]["decoder_rank"]
            return summary[f"{method.split('_')[0]}-rank{rank}-seed{seed}"]
        kind = "learned" if method.startswith("learned") else "pca"
        return summary[f"{kind}-rank{method[len(kind):]}-seed{seed}"]

    def avg(seed, method, mode, split, query, field):
        cells = get(seed, method)["modes"][mode]
        values = [v[field] for key, v in cells.items() if key.endswith(f"/{split}/{query}")]
        if len(values) != 3:
            raise ValueError("Expected all three tasks")
        return float(np.mean(values))

    def accuracy(method, split):
        return [avg(s, method, "replacement", split, "target", "accuracy") for s in range(3)]

    def role_accuracy(method, split, query):
        return float(np.mean([
            avg(s, method, "role_swap", split, query, "accuracy") for s in range(3)
        ]))

    compressed = accuracy("compressed", "test_combinations")
    tpr = accuracy("tpr", "test_combinations")
    native_accuracy = [cell["accuracy"] for report in forges
                       for key, cell in report["independent_forge"].items()
                       if key.endswith("/target")]
    projection_accuracy = [cell["accuracy"] for report in forges
                           for key, cell in report["same_basis_projection"].items()
                           if key.endswith("/target")]

    lines = [
        "# Binding-preservation benchmark: empirical report", "",
        f"Host: `{protocol['model']}` at `{protocol['model_revision']}`.",
        "Three fitting seeds; final partitions were withheld from all fitting and selection.",
        "This is a final-residual readout experiment, followed by an independent whole-model forge.",
        "It does not identify the upstream algorithm that creates bindings.", "",
        "## Main findings", "",
        f"- Co-firing compression retains {np.mean(compressed):.1%} combination accuracy, versus "
        f"{np.mean(accuracy('sae', 'test_combinations')):.1%} for SAE reconstruction and "
        f"{np.mean(accuracy('pca64', 'test_combinations')):.1%} for rank-64 PCA.",
        f"- Supervised TPR combination accuracy ranges from {min(tpr):.1%} to {max(tpr):.1%} "
        "across seeds. It does not establish a robust advantage over the linear controls.",
        f"- Across both final partitions, same-basis projection retains {np.mean(projection_accuracy):.1%} "
        f"accuracy. Independent native execution averages {np.mean(native_accuracy):.1%}, essentially "
        "the 33.3% three-choice chance baseline; its pair accuracy is 0.000 and its KL is 11–14 nats. "
        "At this aggregate level it is indistinguishable from chance, with destroyed output distributions, "
        "not as retaining the decision. The gap from projection is the finding.", "",
        "### Role-swap specificity (task-macro means)", "",
        "Target accuracy is scored against the counterfactual host; unrelated accuracy is scored against",
        "the original host. The latter remains 1.000, while target performance shows whether the edit",
        "changes the queried binding.",
        "| Method | Combination target | Combination unrelated | Template target | Template unrelated |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("sae", "compressed", "tpr", "pca64", "learned32"):
        lines.append(
            f"| {method} | {role_accuracy(method, 'test_combinations', 'target'):.3f} "
            f"| {role_accuracy(method, 'test_combinations', 'unrelated'):.3f} "
            f"| {role_accuracy(method, 'test_templates', 'target'):.3f} "
            f"| {role_accuracy(method, 'test_templates', 'unrelated'):.3f} |"
        )
    lines.extend([
        "", "## Host selection and protocol amendment", "",
        "The initial Qwen2.5-0.5B-Instruct screen was retained as a failed host-selection run.",
        host_amendment["reason"],
        "It was dropped before final capture and fitting; the frozen Qwen2.5-1.5B-Instruct host",
        "passed all three development gates. This decision and the independent SAE decoder-norm",
        "amendment are recorded in `runs/binding/protocol-amendments.json`; neither changes",
        "final-test tuning.", "",
        "## Host competence on final partitions", "",
        "Accuracy uses three candidate answers. Intervals resample the 48 minimal pairs in each cell.",
        "| Task / partition | Accuracy | Pair-bootstrap 95% interval | Both members |",
        "|---|---:|---:|---:|",
    ])
    for key, cell in summaries[0]["identity"]["modes"]["replacement"].items():
        if key.endswith("/target"):
            lo, hi = cell["accuracy_ci95_pair_bootstrap"]
            lines.append(f"| {key.removesuffix('/target')} | {cell['accuracy']:.3f} "
                         f"| {lo:.3f}–{hi:.3f} | {cell['pair_accuracy']:.3f} |")
    lines.extend([
        "", "## Capacity and fitting", "",
        "Fits use 2,304 training and 576 validation query records. Optimized methods have a fixed",
        "1,200-step budget, with checkpoint selection by validation objective every 50 steps,",
        "including step zero. PCA uses training rows only. No final test selects a fit.", "",
        "| Method | Decoder rank (seeds 0/1/2) | Gradient-fitted parameters (seed 0) | Stored scalars (seed 0) |",
        "|---|---|---:|---:|",
    ])
    for method in methods[1:]:
        caps = [get(s, method)["capacity"] for s in range(3)]
        ranks = "/".join(str(c["decoder_rank"]) for c in caps)
        lines.append(f"| {method} | {ranks} | {caps[0]['fitted_parameters']:,} "
                     f"| {caps[0]['stored_scalars']:,} |")
    lines.extend([
        "", "The SAE has unit-norm decoder directions. Compression preserves zeroed rows on disk;",
        "its metadata also reports active scalars. Requested targets 32 and 64 yielded identical",
        "checkpoints within every seed; they are one compression result per seed, not two independent",
        "budget points. Matched controls use the actual compressed decoder ranks.", "",
        "Mean active SAE latents on training records (seeds 0/1/2): "
        + "/".join(f"{sparsity[str(s)]['mean_active_latents']:.2f}" for s in range(3))
        + " of 128; no dead latents. This is moderate sparsity, not a single-feature code.", "",
        "At ranks 64 and above, validation selected the learned control's step-zero PCA initialization.",
        "Those matching results are not independent improvements from learned optimization.", "",
        "## Replacement", "",
        "Numbers are task-macro means, then means across seeds. Full per-task results and pair-bootstrap",
        "intervals remain in each seed's JSON. Seed ranges below are not confidence intervals.",
    ])
    for split in ("test_combinations", "test_templates"):
        lines.extend(["", f"### {split}", "",
                      "| Method | Accuracy | Both members | Host argmax agreement | KL (nats) | Accuracy seed range |",
                      "|---|---:|---:|---:|---:|---:|"])
        for method in methods:
            columns = [
                [avg(s, method, "replacement", split, "target", field) for s in range(3)]
                for field in ("accuracy", "pair_accuracy", "argmax_agreement", "mean_kl_nats")
            ]
            a, p, g, kl = [np.mean(c) for c in columns]
            lines.append(f"| {method} | {a:.3f} | {p:.3f} | {g:.3f} | {kl:.4f} "
                         f"| {min(columns[0]):.3f}–{max(columns[0]):.3f} |")
    lines.extend(["", "## Role edits and unrelated bindings", "",
                  "TPR edits use input parse annotations only. Other methods use counterfactual",
                  "activation donors; those results do not establish autonomous semantic editing.",
                  "Target edits are compared with the counterfactual host, unrelated edits with the original host."])
    for split in ("test_combinations", "test_templates"):
        lines.extend(["", f"### {split}", "",
                      "| Method | Edited target accuracy | Unrelated accuracy | Unrelated KL to original host |",
                      "|---|---:|---:|---:|"])
        for method in methods:
            target = np.mean([avg(s, method, "role_swap", split, "target", "accuracy")
                              for s in range(3)])
            unrelated = np.mean([avg(s, method, "role_swap", split, "unrelated", "accuracy")
                                 for s in range(3)])
            kl = np.mean([avg(s, method, "role_swap", split, "unrelated", "mean_kl_nats")
                          for s in range(3)])
            lines.append(f"| {method} | {target:.3f} | {unrelated:.3f} | {kl:.4f} |")
    lines.extend(["", "## Independent execution", "",
                  "Native models were saved and reloaded, then run on token IDs without host activation",
                  "hooks. RoPE is preserved; tied host weights are numerically unchanged but untied",
                  "before projection. No fine-tuning or host-wrapped fallback is used.", "",
                  "| Seed | Actual basis width | Split | Projection accuracy | Native accuracy | Projection KL | Native KL |",
                  "|---:|---:|---|---:|---:|---:|---:|"])
    for seed, report in enumerate(forges):
        for split in ("test_combinations", "test_templates"):
            means = []
            for mode in ("same_basis_projection", "independent_forge"):
                cells = [v for key, v in report[mode].items() if key.endswith(f"/{split}/target")]
                means.append((np.mean([v["accuracy"] for v in cells]),
                              np.mean([v["mean_kl_nats"] for v in cells])))
            lines.append(f"| {seed} | {report['build']['basis_width']} | {split} "
                         f"| {means[0][0]:.3f} | {means[1][0]:.3f} "
                         f"| {means[0][1]:.4f} | {means[1][1]:.4f} |")
    lines.extend(["", "## Interpretation limits", "",
                  "- Candidate accuracy is discrimination among three alternatives, not free generation.",
                  "- Host failures are retained; conditional-on-host-correct scores are in the JSON reports.",
                  "- TPR supervision and donor-assisted baselines have different information budgets.",
                  "- No annotation-matched additive or atomic-pair control was run; a TPR advantage",
                  "  would not isolate the benefit of tensor-product factorization.",
                  "- Equal decoder rank does not equal parameter count, sparsity, or supervision.",
                  "- A single final-layer SAE basis used across all layers can fail because the layers need",
                  "  different representations and nonlinear operations do not commute with projection.",
                  "- Fitted subspaces and negative outcomes are not global optima or irreducibility proofs.",
                  "- Holdouts apply to benchmark fitting, not to the host's pretraining history.",
                  "- The native 33.7% aggregate is interpreted against the 33.3% three-choice chance",
                  "  baseline; it is not described as retained decision accuracy.",
                  "", "## Relation to the paper and sibling projects", "",
                  "[DISCOVER](https://arxiv.org/pdf/2608.29530) motivates replacement and structured",
                  "intervention as stronger evidence than representational similarity. Here the role",
                  "assignments are supplied, and only a final question-conditioned residual is modeled.",
                  "This study neither replicates the paper's complete experiments nor discovers unique",
                  "internal symbolic roles.", "",
                  "For Polygram, the negative compression result concerns this co-firing merge configuration.",
                  "It does not evaluate quantum encodings or establish that every compression strategy fails.",
                  "Co-firing and similar scalar ablation KLs remain insufficient evidence of substitutability.",
                  "For lm-sae and sae-forge, the projection/execution comparison supports separating retained",
                  "readout information from the ability to execute the computation in that representation.",
                  "It supplies a bounded example, not a theorem about an intrinsic SAE composition tax.",
                  "The already retracted writer-output preservation claim in",
                  "[sae-forge's research note](../../../sae-forge/docs/two_basis_forge.md) is not used as evidence.",
                  "", "## Reproduction", "",
                  "See `examples/binding_benchmark/README.md`, `runs/binding/protocol.json`,",
                  "`runs/binding/protocol-amendments.json`, the recorded",
                  "protocol amendment, frozen fit hashes, per-example JSONL files and three forge build reports.",
                  "Large binary checkpoints are saved locally and gitignored.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/binding"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(args.root))


if __name__ == "__main__":
    main()
