"""Common causal measurements and pair-clustered uncertainty."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .capture import tail_logits
from .representations import reconstruct


def verify_frozen_fits(directory):
    manifest = json.loads((directory / "frozen.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Fit artifact changed after freezing: {name}")


def counterpart_indices(records):
    lookup = {(r["pair_id"], r["member"], r["unrelated"]): i for i, r in enumerate(records)}
    if len(lookup) != len(records):
        raise ValueError("Duplicate pair/member/query record")
    return torch.tensor([lookup[r["pair_id"], 1 - r["member"], r["unrelated"]]
                         for r in records])


def pair_interval(values, seed=0, repetitions=2000):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return None
    rng = np.random.default_rng(seed)
    boot = a[rng.integers(0, len(a), (repetitions, len(a)))].mean(1)
    return np.quantile(boot, [0.025, 0.975]).tolist()


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["task"], row["split"], row["unrelated"])].append(row)
    report = {}
    for (task, split, unrelated), group in grouped.items():
        pairs = defaultdict(list)
        for row in group:
            pairs[row["pair_id"]].append(row)
        if any(len(pair) != 2 for pair in pairs.values()):
            raise ValueError("Metrics require complete minimal pairs")
        pair_means = [np.mean([r["correct"] for r in p]) for p in pairs.values()]
        pair_correct = [all(r["correct"] for r in p) for p in pairs.values()]
        eligible = [r for r in group if r["reference_correct"]]
        report[f"{task}/{split}/{'unrelated' if unrelated else 'target'}"] = {
            "n": len(group), "n_pairs": len(pairs),
            "accuracy": float(np.mean(pair_means)),
            "accuracy_ci95_pair_bootstrap": pair_interval(pair_means),
            "pair_accuracy": float(np.mean(pair_correct)),
            "pair_accuracy_ci95": pair_interval(pair_correct),
            "reference_accuracy": float(np.mean([r["reference_correct"] for r in group])),
            "accuracy_on_reference_correct": float(np.mean([r["correct"] for r in eligible]))
            if eligible else None,
            "n_reference_correct": len(eligible),
            "candidate_agreement": float(np.mean([r["candidate_agreement"] for r in group])),
            "argmax_agreement": float(np.mean([r["argmax_agreement"] for r in group])),
            "mean_kl_nats": float(np.mean([r["kl_nats"] for r in group])),
            "mean_margin_nats": float(np.mean([r["margin_nats"] for r in group])),
        }
    return report


def compare_logits(logits, reference, records):
    if not torch.isfinite(logits).all() or not torch.isfinite(reference).all():
        raise FloatingPointError("Non-finite logits cannot be scored")
    lp, lq = reference.float().log_softmax(-1), logits.float().log_softmax(-1)
    kl = (lp.exp() * (lp - lq)).sum(-1)
    rows = []
    for i, rec in enumerate(records):
        ids = rec["candidate_ids"]
        j = int(logits[i, ids].argmax())
        refj = int(reference[i, ids].argmax())
        target = rec["candidates"].index(rec["target"])
        alternatives = [k for k in range(len(ids)) if k != target]
        rows.append({
            "id": rec["id"], "pair_id": rec["pair_id"], "task": rec["task"],
            "split": rec["split"], "unrelated": rec["unrelated"],
            "target": rec["target"], "prediction": rec["candidates"][j],
            "correct": j == target, "reference_correct": refj == target,
            "candidate_agreement": j == refj,
            "argmax_agreement": int(logits[i].argmax()) == int(reference[i].argmax()),
            "kl_nats": float(kl[i]),
            "margin_nats": float(logits[i, ids[target]] - logits[i, [ids[k] for k in alternatives]].max()),
        })
    return rows


def evaluate_hidden(h, reconstructed, records, tail, mode, batch_size=64):
    cf = counterpart_indices(records)
    if mode == "replacement":
        edited, reference, targets = reconstructed, h, records
    elif mode == "role_swap":
        edited = h + reconstructed[cf] - reconstructed
        unrelated = torch.tensor([r["unrelated"] for r in records])[:, None]
        reference = torch.where(unrelated, h, h[cf])
        targets = [{**r, "target": records[j]["target"]} for r, j in zip(records, cf.tolist())]
    elif mode == "no_op":
        edited, reference, targets = h + (reconstructed - reconstructed), h, records
    else:
        raise ValueError(mode)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(h), batch_size):
            end = start + batch_size
            logits = tail_logits(edited[start:end], tail)
            host = tail_logits(reference[start:end], tail)
            rows.extend(compare_logits(logits, host, targets[start:end]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--fits", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = json.loads((args.capture / "capture.json").read_text())
    fits = json.loads(args.fits.read_text())
    if info["protocol"] != fits["protocol"]:
        raise ValueError("Fit and evaluation protocols differ")
    torch.set_num_threads(info["protocol"]["threads"])
    h = torch.from_numpy(np.load(args.capture / "hidden.npy"))
    records = json.loads((args.capture / "records.json").read_text())
    if any(r["split"].startswith("test_") for r in records):
        verify_frozen_fits(args.fits.parent)
    tail = torch.load(args.capture / "tail.pt", weights_only=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    no_op_rows = None
    checkpoint_hashes = {}
    for meta in [{"kind": "identity", "artifact": "identity"}] + fits["representations"]:
        artifact = meta["artifact"]
        name = Path(artifact).stem
        if str(artifact).endswith(".safetensors"):
            digest = hashlib.sha256(Path(artifact).read_bytes()).hexdigest()
            if digest in checkpoint_hashes:
                previous = checkpoint_hashes[digest]
                summary[name] = {**summary[previous], "capacity": meta, "identical_to": previous}
                continue
            checkpoint_hashes[digest] = name
        reconstruction = h if artifact == "identity" else reconstruct(artifact, h, records)
        summary[name] = {"capacity": meta, "modes": {}}
        for mode in ("replacement", "role_swap", "no_op"):
            print(f"evaluating {name} {mode}", flush=True)
            if mode == "no_op" and no_op_rows is not None:
                if not torch.equal(h + (reconstruction - reconstruction), h):
                    raise FloatingPointError("No-op edit did not preserve the residual exactly")
                rows = no_op_rows
            else:
                rows = evaluate_hidden(h, reconstruction, records, tail, mode)
                if mode == "no_op":
                    no_op_rows = rows
            (args.output / f"{name}-{mode}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            summary[name]["modes"][mode] = aggregate(rows)
        summary[name]["edit_information"] = (
            "semantic annotations only; no counterfactual activation donor"
            if meta["kind"] == "tpr" else "counterfactual activation donor required"
        )
        summary[name]["role_swap_reference"] = (
            "counterfactual host on target question; original host on unrelated question"
        )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
