"""Capture the final pre-normalization residual, with an identity readout check."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .data import SYSTEM, read_dataset
from .host import load_host


def tail_logits(h, tail):
    import torch

    x = h.to(tail["head"].dtype)
    normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True)
                                        + tail["eps"])).to(x.dtype)
    return torch.nn.functional.linear(normalized * tail["norm"], tail["head"]).float()


def capture(model, tokenizer, examples, batch_size, output):
    import torch

    if model.config.model_type != "qwen2":
        raise ValueError("Capture protocol is currently validated only for Qwen2")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tail = {"head": model.lm_head.weight.detach().cpu(),
            "norm": model.model.norm.weight.detach().cpu(),
            "eps": model.config.rms_norm_eps}
    captured = []

    def hook(module, args):
        captured.append(args[0][:, -1].detach().float().cpu())

    handle = model.model.norm.register_forward_pre_hook(hook)
    records = [{**asdict(ex), "unrelated": unrelated,
                "target": ex.unrelated_answer if unrelated else ex.answer}
               for ex in examples for unrelated in (False, True)]
    prompts = [ex.prompt(unrelated) for ex in examples for unrelated in (False, True)]
    hidden, max_error, identity_kls, identity_agreement = [], 0.0, [], []
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                batch = records[start:start + batch_size]
                texts = [tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
                    tokenize=False, add_generation_prompt=True,
                ) for prompt in prompts[start:start + batch_size]]
                inputs = tokenizer(texts, padding=True, return_tensors="pt", add_special_tokens=False)
                inputs["position_ids"] = (inputs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
                captured.clear()
                out = model(**inputs, use_cache=False, logits_to_keep=1)
                if len(captured) != 1:
                    raise RuntimeError("Expected exactly one final norm call")
                h = captured[0]
                logits = tail_logits(h, tail)
                error = float((logits - out.logits[:, -1].float()).abs().max())
                max_error = max(max_error, error)
                lp = out.logits[:, -1].float().log_softmax(-1)
                lq = logits.log_softmax(-1)
                kl = (lp.exp() * (lp - lq)).sum(-1)
                identity_kls.extend(kl.tolist())
                identity_agreement.extend((logits.argmax(-1) == lp.argmax(-1)).tolist())
                if float(kl.max()) > 1e-3:
                    raise RuntimeError(f"Identity readout KL exceeds numerical tolerance: {kl.max()}")
                hidden.append(h.numpy())
                for rec, mask in zip(batch, inputs["attention_mask"]):
                    rec["prompt_tokens"] = int(mask.sum())
                    tokens = [tokenizer.encode(c, add_special_tokens=False) for c in rec["candidates"]]
                    if any(len(t) != 1 for t in tokens):
                        raise ValueError("Frozen final-residual protocol requires single-token answers")
                    rec["candidate_ids"] = [t[0] for t in tokens]
                if start % (batch_size * 10) == 0:
                    print(f"captured {start + len(batch)}/{len(records)}", flush=True)
    finally:
        handle.remove()
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "hidden.npy", np.concatenate(hidden))
    (output / "records.json").write_text(json.dumps(records) + "\n")
    torch.save(tail, output / "tail.pt")
    return {"n_records": len(records), "d_model": hidden[0].shape[1],
            "identity_max_logit_error": max_error,
            "identity_mean_kl": float(np.mean(identity_kls)),
            "identity_max_kl": float(np.max(identity_kls)),
            "identity_argmax_agreement": float(np.mean(identity_agreement)),
            "location": "final residual before model.norm, last prompt position",
            "remaining_computation": "RMSNorm and lm_head only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    path = Path(protocol["dataset_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != protocol["dataset_sha256"]:
        raise ValueError("Dataset has changed since protocol freeze")
    splits = args.splits.split(",")
    examples = [ex for ex in read_dataset(path) if ex.split in splits]
    if not examples:
        raise ValueError("No examples selected")
    started = time.time()
    model, tok = load_host(protocol["model"], protocol["threads"], protocol["dtype"])
    if model.config._commit_hash != protocol["model_revision"]:
        raise ValueError("Host revision differs from frozen protocol")
    report = capture(model, tok, examples, args.batch_size, args.output)
    report.update(protocol=protocol, splits=splits, elapsed_seconds=time.time() - started)
    (args.output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
