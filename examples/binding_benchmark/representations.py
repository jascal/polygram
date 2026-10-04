"""Fitting methods for the preregistered final-residual comparison."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class SparseAE(nn.Module):
    def __init__(self, d, width):
        super().__init__()
        self.encoder = nn.Linear(d, width)
        self.decoder = nn.Linear(width, d, bias=False)
        nn.init.zeros_(self.encoder.bias)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12))

    def encode(self, x):
        return self.encoder(x).relu()

    def forward(self, x, symbols=None):
        return self.decoder(self.encode(x))


class Subspace(nn.Module):
    def __init__(self, basis):
        super().__init__()
        self.basis = nn.Parameter(basis.clone())

    def forward(self, x, symbols=None):
        q = torch.linalg.qr(self.basis).Q
        return x @ q @ q.T


class RoleFiller(nn.Module):
    def __init__(self, d, fillers, roles, filler_dim=16, role_dim=4):
        super().__init__()
        self.fillers = nn.Embedding(fillers, filler_dim)
        self.roles = nn.Embedding(roles, role_dim)
        self.output = nn.Linear(filler_dim * role_dim, d)
        nn.init.normal_(self.fillers.weight, std=0.2)
        nn.init.normal_(self.roles.weight, std=0.2)

    def forward(self, x, symbols):
        f = self.fillers(symbols[:, :, 0])
        r = self.roles(symbols[:, :, 1])
        product = torch.einsum("bif,bir->bfr", f, r).flatten(1)
        return self.output(product)


def symbol_vocab(records):
    fillers = sorted({f"{row['task']}:{f}" for row in records for _, f in row["bindings"]})
    roles = sorted({f"{row['task']}:{r}:{row['unrelated']}"
                    for row in records for r, _ in row["bindings"]})
    return {"fillers": {f: i for i, f in enumerate(fillers)},
            "roles": {r: i for i, r in enumerate(roles)}}


def symbols_for(records, vocab):
    return torch.tensor([
        [(vocab["fillers"][f"{row['task']}:{f}"],
          vocab["roles"][f"{row['task']}:{r}:{row['unrelated']}"])
         for r, f in row["bindings"]] for row in records
    ])


def candidate_logits(x, tail):
    normalized = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + tail["eps"])
    return (normalized * tail["norm"].float()) @ tail["candidate_head"].T


def fit(model, x, symbols, train, val, protocol, kind, mean, scale, tail, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol["learning_rate"])
    rng = torch.Generator().manual_seed(seed)
    history, best, best_state = [], float("inf"), None

    def objective(indices):
        prediction = model(x[indices], symbols[indices])
        mse = (prediction - x[indices]).square().mean()
        loss = mse
        if kind == "sae":
            loss = loss + protocol["sae_l1"] * model.encode(x[indices]).mean()
        elif kind == "learned":
            p = candidate_logits(x[indices] * scale + mean, tail).softmax(-1)
            lq = candidate_logits(prediction * scale + mean, tail).log_softmax(-1)
            loss = loss + (p * (p.clamp_min(1e-12).log() - lq)).sum(-1).mean()
        return loss

    for step in range(protocol["steps"] + 1):
        if step % protocol["check_every"] == 0:
            with torch.no_grad():
                score = float(objective(val))
            if not np.isfinite(score):
                raise FloatingPointError(f"Non-finite {kind} validation objective at step {step}")
            history.append({"step": step, "validation_objective": score})
            if score < best:
                best, best_state = score, copy.deepcopy(model.state_dict())
        if step == protocol["steps"]:
            break
        indices = train[torch.randint(len(train), (protocol["batch_size"],), generator=rng)]
        optimizer.zero_grad()
        loss = objective(indices)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if kind == "sae":
            model.normalize_decoder()
    model.load_state_dict(best_state)
    return history


def save_fit(path, model, kind, mean, scale, vocab, history, seed):
    torch.save({"kind": kind, "state": model.state_dict(), "mean": mean, "scale": scale,
                "vocab": vocab, "seed": seed}, path)
    if kind == "sae":
        decoder = model.decoder.weight.detach().T
        latent_width = model.encoder.out_features
    elif kind == "tpr":
        decoder = model.output.weight.detach().T
        latent_width = decoder.shape[0]
    else:
        decoder = model.basis.detach().T
        latent_width = decoder.shape[0]
    rank = int(torch.linalg.matrix_rank(decoder))
    params = sum(p.numel() for p in model.parameters())
    report = {"kind": kind, "seed": seed, "decoder_rank": rank,
              "latent_width": latent_width, "fitted_parameters": params,
              "stored_scalars": params + mean.numel() + scale.numel(),
              "history": history, "artifact": str(path)}
    if kind == "sae":
        report["decoder_constraint"] = "unit L2 norm per decoder direction after every update"
    path.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def load_fit(path):
    saved = torch.load(path, weights_only=True)
    state, kind = saved["state"], saved["kind"]
    if kind == "sae":
        width, d = state["encoder.weight"].shape
        model = SparseAE(d, width)
    elif kind == "tpr":
        nf, df = state["fillers.weight"].shape
        nr, dr = state["roles.weight"].shape
        model = RoleFiller(state["output.weight"].shape[0], nf, nr, df, dr)
    elif kind in ("pca", "learned"):
        model = Subspace(state["basis"])
    else:
        raise ValueError(kind)
    model.load_state_dict(state)
    return model.eval(), saved


def reconstruct(path, h, records):
    if str(path).endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(str(path))
        return ((h - state["b_dec"]) @ state["W_enc"] + state["b_enc"]).relu() \
            @ state["W_dec"] + state["b_dec"]
    model, saved = load_fit(path)
    x = (h - saved["mean"]) / saved["scale"]
    syms = symbols_for(records, saved["vocab"])
    with torch.no_grad():
        return model(x, syms) * saved["scale"] + saved["mean"]


def compress_sae(model, mean, scale, xtrain, output, protocol, seed):
    from safetensors.numpy import load_file, save_file

    from polygram.behavioural.report import CandidatePair, ValidationReport, ValidationSummary
    from polygram.compression import Compressor
    from polygram.config import CompressionConfig

    state = {
        "W_enc": model.encoder.weight.detach().T.numpy().copy() / float(scale),
        "W_dec": model.decoder.weight.detach().T.numpy().copy() * float(scale),
        "b_enc": model.encoder.bias.detach().numpy().copy(), "b_dec": mean.numpy().copy(),
    }
    source = output / f"sae-seed{seed}.safetensors"
    save_file(state, str(source))
    with torch.no_grad():
        acts = model.encode(xtrain).numpy()
    fires = (acts > 0).astype(np.int64)
    counts = fires.sum(0)
    both = fires.T @ fires
    dec = state["W_dec"]
    normalized = dec / np.maximum(np.linalg.norm(dec, axis=1, keepdims=True), 1e-12)
    cos2 = (normalized @ normalized.T) ** 2
    pairs = []
    for i in range(len(dec)):
        for j in range(i + 1, len(dec)):
            either = int(counts[i] + counts[j] - both[i, j])
            pairs.append(CandidatePair(
                i=i, j=j, polygram_overlap=float("nan"), decoder_overlap=float(cos2[i, j]),
                jaccard=float(both[i, j] / max(either, 1)), pearson_activation=float("nan"),
                kl_ablate_i=float("nan"), kl_ablate_j=float("nan"),
                kl_ratio_paired=float("nan"), kl_log_ratio_abs=float("nan"),
                n_fires_i=int(counts[i]), n_fires_j=int(counts[j]),
                n_both_fire=int(both[i, j]), n_either_fire=either, gate_pass=False,
            ))
    summary = ValidationSummary(float("nan"), float("nan"), float("nan"), float("nan"),
                                float("nan"), {}, "training co-firing only; no causal confirmation")
    report = ValidationReport(
        schema_version=1, dictionary_name="binding_final_residual", model_name=protocol["model"],
        layer=-1, n_prompts=len(xtrain), n_tokens=len(xtrain),
        polygram_overlap_threshold=1.0, jaccard_threshold=1.0, min_firing_rate=0,
        min_both_fire=1, feature_ids=tuple(range(len(dec))), pairs=tuple(pairs),
        summary=summary, confirmed=(),
    )
    report.to_json(output / f"compression-input-seed{seed}.json")
    compressor = Compressor(report, source, config=CompressionConfig(
        strategy="merge", rep_selection="n_fires", score_field="jaccard",
    ))
    results = []
    for target in protocol["compression_target_widths"]:
        dest = output / f"compressed-target{target}-seed{seed}.safetensors"
        result = compressor.apply(compressor.plan_with_target(target), output_checkpoint=dest)
        result.report.to_json(dest.with_suffix(".compression_report.json"))
        compressed = load_file(str(dest))
        kept = np.linalg.norm(compressed["W_dec"], axis=1) > 1e-12
        rank = int(np.linalg.matrix_rank(compressed["W_dec"][kept]))
        meta = {"kind": "polygram_compressed", "seed": seed, "target_width": target,
                "latent_width": int(kept.sum()), "decoder_rank": rank,
                "artifact": str(dest),
                "fitted_parameters": sum(p.numel() for p in model.parameters()),
                "stored_scalars": sum(v.size for v in compressed.values()),
                "active_scalars": int(kept.sum()) * (2 * dec.shape[1] + 1) + dec.shape[1],
                "ranking": "training co-firing Jaccard; Polygram target-K merge",
                "representative_selection": "n_fires (no ablation-KL available)",
                "causal_confirmation": False}
        dest.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
        results.append(meta)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    protocol = json.loads((args.capture / "capture.json").read_text())["protocol"]
    torch.set_num_threads(protocol["threads"])
    torch.manual_seed(args.seed)
    h = torch.from_numpy(np.load(args.capture / "hidden.npy"))
    records = json.loads((args.capture / "records.json").read_text())
    if any(r["split"] not in ("train", "validation") for r in records):
        raise ValueError("Fit capture contains final-test data")
    train = torch.tensor([i for i, r in enumerate(records) if r["split"] == "train"])
    val = torch.tensor([i for i, r in enumerate(records) if r["split"] == "validation"])
    mean = h[train].mean(0)
    scale = (h[train] - mean).square().mean().sqrt()
    x = (h - mean) / scale
    vocab = symbol_vocab([records[i] for i in train.tolist()])
    symbols = symbols_for(records, vocab)
    tail = torch.load(args.capture / "tail.pt", weights_only=True)
    ids = sorted({i for index in train.tolist() for i in records[index]["candidate_ids"]})
    tail["candidate_head"] = tail.pop("head")[ids].float()
    args.output.mkdir(parents=True, exist_ok=True)
    specs = []
    sae = SparseAE(h.shape[1], protocol["sae_width"])
    print("fitting SAE", flush=True)
    history = fit(sae, x, symbols, train, val, protocol, "sae", mean, scale, tail, args.seed)
    specs.append(save_fit(args.output / f"sae-seed{args.seed}.pt", sae, "sae", mean,
                          scale, vocab, history, args.seed))
    print("compressing SAE with Polygram", flush=True)
    specs.extend(compress_sae(sae, mean, scale, x[train], args.output, protocol, args.seed))
    tpr = RoleFiller(h.shape[1], len(vocab["fillers"]), len(vocab["roles"]),
                     protocol["tpr_filler_dim"], protocol["tpr_role_dim"])
    print("fitting role-filler model", flush=True)
    history = fit(tpr, x, symbols, train, val, protocol, "tpr", mean, scale, tail, args.seed)
    specs.append(save_fit(args.output / f"tpr-seed{args.seed}.pt", tpr, "tpr", mean,
                          scale, vocab, history, args.seed))
    ranks = sorted(set(protocol["control_ranks"]) | {r["decoder_rank"] for r in specs})
    _, _, v = torch.linalg.svd(x[train], full_matrices=False)
    for rank in ranks:
        for kind in ("pca", "learned"):
            print(f"fitting {kind} rank {rank}", flush=True)
            sub = Subspace(v[:rank].T)
            history = [] if kind == "pca" else fit(
                sub, x, symbols, train, val, protocol, kind, mean, scale, tail, args.seed,
            )
            meta = save_fit(args.output / f"{kind}-rank{rank}-seed{args.seed}.pt", sub,
                            kind, mean, scale, vocab, history, args.seed)
            if kind == "pca":
                meta["fitted_parameters"] = 0
                Path(meta["artifact"]).with_suffix(".json").write_text(json.dumps(meta, indent=2))
            specs.append(meta)
    (args.output / f"fits-seed{args.seed}.json").write_text(json.dumps({
        "protocol": protocol, "train_records": len(train), "validation_records": len(val),
        "representations": specs,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
