"""Independent sae-forge execution, with a same-basis activation control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from .capture import tail_logits
from .data import SYSTEM
from .evaluate import aggregate, compare_logits, evaluate_hidden, verify_frozen_fits
from .host import load_host


def native_config(adapter, host, width):
    config = adapter.build_native_config(host, width, attention_width="host")
    # Transformers 5 moved theta into rope_parameters; the installed adapter reads the old field.
    rope = getattr(host.config, "rope_parameters", None)
    if rope is not None:
        if rope.get("rope_type", "default") != "default":
            raise ValueError("This benchmark supports default RoPE only")
        config.rope_theta = float(rope["rope_theta"])
    config.forward_mode = "native_in_basis"
    return config


def untie_for_projection(host):
    was_tied = bool(host.config.tie_word_embeddings)
    if was_tied:
        # Embed @ pinv(D) and unembed @ D.T differ in a non-orthogonal SAE basis.
        host.lm_head.weight = torch.nn.Parameter(host.lm_head.weight.detach().clone(),
                                               requires_grad=False)
        host.config.tie_word_embeddings = False
    return was_tied


def build(protocol, checkpoint, output):
    import saeforge
    from saeforge import FeatureBasis, SubspaceProjector
    from saeforge.adapters import adapter_for
    from saeforge.model import NativeModel

    basis = FeatureBasis.from_polygram_checkpoint(checkpoint)
    host, tokenizer = load_host(protocol["model"], protocol["threads"], protocol["dtype"])
    if host.config._commit_hash != protocol["model_revision"]:
        raise ValueError("Host revision mismatch")
    was_tied = untie_for_projection(host)
    adapter = adapter_for(host)
    projector = SubspaceProjector(basis, scale_boost=1.0)
    weights = projector.project_module(host, attention_width="host")
    config = native_config(adapter, host, basis.n_features)
    native = NativeModel.from_projected_weights(config, weights)
    expected = set(native.torch_module.state_dict())
    allowed_missing = {"lm_head.weight"} if config.tied_embeddings else set()
    missing = expected - set(weights) - allowed_missing
    if missing:
        raise ValueError(f"Forge has uninitialized state: {sorted(missing)}")
    native.resolved_forward_mode = "native_in_basis"
    native.save_pretrained(output)
    metadata = {
        "saeforge_version": saeforge.__version__, "saeforge_source": str(Path(saeforge.__file__)),
        "basis_checkpoint": str(checkpoint), "basis_width": basis.n_features,
        "basis_checkpoint_sha256": hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        "basis_rank": int(np.linalg.matrix_rank(basis.W_dec)),
        "host_revision": protocol["model_revision"], "forward_mode": config.forward_mode,
        "attention_width": "host", "scale_boost": 1.0,
        "rope_theta": config.rope_theta,
        "host_embeddings_untied_for_projection": was_tied,
        "parameters": sum(p.numel() for p in native.torch_module.parameters()),
        "finetuning_steps": 0,
    }
    package = Path(saeforge.__file__).resolve().parents[1]
    metadata["saeforge_revision"] = subprocess.check_output(
        ["git", "-C", str(package), "rev-parse", "HEAD"], text=True,
    ).strip()
    (output / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    del native, host, weights, projector
    gc.collect()
    return basis, tokenizer, metadata


def evaluate_independent(model_dir, tokenizer, records, h, tail):
    from saeforge.model import NativeModel

    # This function reloads weights from disk; neither host nor activations enter the forward pass.
    native = NativeModel.load_pretrained(model_dir)
    if native.config.forward_mode != "native_in_basis":
        raise ValueError("Host-wrapped models are not independent native forges")
    module = native.torch_module.eval()
    if any(m._forward_hooks or m._forward_pre_hooks for m in module.modules()):
        raise ValueError("Independent model unexpectedly has activation hooks")
    rows = []
    with torch.inference_mode():
        # No padding: native positions and attention masks cannot silently differ from the host.
        for i, row in enumerate(records):
            question = row["unrelated_question"] if row["unrelated"] else row["question"]
            text = f"{row['context']}\n{question}"
            ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                tokenize=True, add_generation_prompt=True,
            )
            if hasattr(ids, "keys"):
                ids = ids["input_ids"]
            out = module(input_ids=torch.tensor([ids]))
            logits = (out if isinstance(out, torch.Tensor) else out.logits)[:, -1].float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite independent forge output at {row['id']}")
            host_logits = tail_logits(h[i:i+1], tail)
            rows.extend(compare_logits(logits, host_logits, [row]))
            if (i + 1) % 24 == 0:
                print(f"forged execution {i + 1}/{len(records)}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    verify_frozen_fits(args.checkpoint.parent)
    protocol = json.loads((args.capture / "capture.json").read_text())["protocol"]
    torch.set_num_threads(protocol["threads"])
    args.output.mkdir(parents=True, exist_ok=True)
    model_dir = args.output / "model"
    if (model_dir / "build.json").exists():
        from saeforge import FeatureBasis
        from transformers import AutoTokenizer

        metadata = json.loads((model_dir / "build.json").read_text())
        digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        if metadata["basis_checkpoint_sha256"] != digest:
            raise ValueError("Saved forge uses a different checkpoint")
        if metadata["host_revision"] != protocol["model_revision"]:
            raise ValueError("Saved forge uses a different host revision")
        basis = FeatureBasis.from_polygram_checkpoint(args.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(protocol["model"], local_files_only=True)
    else:
        basis, tokenizer, metadata = build(protocol, args.checkpoint, model_dir)
    if args.build_only:
        print(json.dumps(metadata, indent=2), flush=True)
        return
    h = torch.from_numpy(np.load(args.capture / "hidden.npy"))
    records = json.loads((args.capture / "records.json").read_text())
    tail = torch.load(args.capture / "tail.pt", weights_only=True)
    d = torch.tensor(basis.W_dec, dtype=torch.float32)
    projected = h @ torch.linalg.pinv(d) @ d
    projection_rows = evaluate_hidden(h, projected, records, tail, "replacement")
    native_rows = evaluate_independent(args.output / "model", tokenizer, records, h, tail)
    for name, rows in (("same_basis_projection", projection_rows), ("independent_forge", native_rows)):
        (args.output / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {"build": metadata, "same_basis_projection": aggregate(projection_rows),
              "independent_forge": aggregate(native_rows)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
