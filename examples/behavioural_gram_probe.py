"""Behavioural-Gram one-pair probe — does Polygram's predicted overlap
line up with real-model co-firing and substitutability?

PR #18 (decoder-Gram validity) settled the *decoder-geometry* validity
question for Polygram's predicted Gram. Its closing caveat names the
next gap explicitly:

    Two SAE features can have orthogonal decoder columns but still
    co-fire on the same inputs (and vice versa). The behavioural-Gram
    comparison would need a forward-pass infrastructure Polygram
    doesn't have.

This spike (per `tech-debt-backlog/tasks.md` §4.2) builds the smallest
possible such infrastructure for *one pair* of features and tests three
real-model statistics against Polygram's predicted Gram entry.

Scope (deliberately narrow):

- One model: GPT-2 small (HuggingFace `gpt2`) at residual stream
  layer 0 (`blocks.0.hook_resid_pre` — the input to the first
  transformer block, after token + position embeddings).
- One SAE: `jbloom/GPT2-Small-SAEs-Reformatted`'s
  `blocks.0.hook_resid_pre` (24576-feature, the same SAE PR #16/PR #18
  used).
- One within-cluster pair: `feat_7836 ↔ feat_11978`
  (Polygram-predicted 0.987, real-decoder 0.992 squared cosine —
  the highest-overlap pair from PR #18's selection).
- One contrast pair: `feat_7836 ↔ feat_15796` (cross-cluster,
  Polygram MPS-predicted 0.464, real-decoder 0.904 — chosen for
  contrast, since on the Real SAE every cross-cluster pair has high
  real overlap due to the projection-similar selection criterion).
- No φ optimization, no Dictionary baking, no Cancellation runs.
  Polygram's φ knob doesn't map to W_dec; this probe deliberately
  doesn't try to invert that gap. It is purely observational.

Three real-model statistics per pair, each compared against Polygram's
predicted overlap and the real-decoder squared cosine:

1. **Co-occurrence rate** — `P(B fires | A fires)` across token
   positions, where "fires" is `f_i > 0` (the SAE's ReLU is the
   natural threshold).
2. **Activation correlation** — Pearson correlation of the two
   features' raw post-ReLU activations across all token positions.
3. **Ablation-KL substitutability** — for each token position where
   feature A fires, run a counterfactual forward pass with A's
   decoder contribution subtracted from the residual stream
   (`x' = x - f_A * W_dec[A, :]`); compute KL divergence between
   the next-token distribution under baseline and ablate-A. Same
   for B. The pair is *substitutable* if `KL(ablate-A) ≈
   KL(ablate-B)` — ablating either has similar downstream effect.

Reproducibility::

    python examples/behavioural_gram_probe.py

The Real SAE fixture is auto-skipped if
`./scratch/real-sae/blocks.0.hook_resid_pre/sae_weights.safetensors`
isn't on disk; see `docs/research/cross-encoding-stability.md` for the
download command. `transformers` and `torch` are imported lazily and
the script exits cleanly with a hint if either is missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from polygram import Dictionary, MPSRung1
from polygram.sae_import import from_sae_lens

# Fixed prompt set — diverse content covering various semantic
# territories so that SAE features get exercised across a range of
# contexts. Each prompt is a short paragraph (3-5 sentences) to give
# meaningful per-prompt token counts. Total tokenizes to ~1500 tokens
# on GPT-2's BPE vocabulary, comfortably above the lower bound where
# Pearson and KL averages stabilize.
PROMPTS: tuple[str, ...] = (
    (
        "The dog wagged its tail and barked at the mail carrier. "
        "Its owner, an elderly woman, called from the porch and "
        "told the puppy to come inside. The mail carrier laughed "
        "and continued down the street, his canvas bag slung over "
        "one shoulder."
    ),
    (
        "Hawks circled high above the meadow, watching for movement. "
        "A field mouse darted from one tuft of grass to another, "
        "unaware of the predator overhead. The eagle had built its "
        "nest on a cliff overlooking the river, and from there it "
        "could see for miles in every direction."
    ),
    (
        "Dr. Smith examined the patient's chart and prescribed a "
        "course of antibiotics. The patient, a 54-year-old woman "
        "with a stubborn sinus infection, had tried three different "
        "remedies before coming to the clinic. The pharmacist "
        "filled the prescription within twenty minutes."
    ),
    (
        "The professor wrote a long equation across the chalkboard "
        "and turned to face the class. Several students were taking "
        "notes furiously while others stared at their laptops. "
        "Quantum mechanics describes the behaviour of particles at "
        "very small scales, where intuitions from classical physics "
        "break down."
    ),
    (
        "She loaded the dishwasher and started the wash cycle before "
        "going to bed. The kitchen still smelled of garlic and "
        "tomato from the dinner she had cooked. Tomorrow she would "
        "wake early, finish a stack of paperwork, and drive across "
        "town to meet a client."
    ),
    (
        "The carpenter measured the board twice before making the "
        "cut. He had been doing this kind of work for thirty years "
        "and the shop was filled with the smell of pine sawdust. "
        "His apprentice watched intently, learning to recognize "
        "where the grain would split if the saw moved too quickly."
    ),
    (
        "Spring rain pattered against the windows of the old "
        "farmhouse, and the family gathered around the fireplace to "
        "read. Outside, geese flew south in a perfect V-formation, "
        "their cries echoing through the cold air. The youngest "
        "child watched them disappear into the gray sky."
    ),
    (
        "Investors watched the stock market closely throughout the "
        "trading day. The price of the company's shares fluctuated "
        "wildly after the surprise earnings report, dropping nearly "
        "ten percent in the morning before rebounding by close. "
        "Several analysts revised their forecasts downward."
    ),
    (
        "The librarian shelved the returned books in alphabetical "
        "order, sorting them first by author and then by title. "
        "A small group of children sat in the reading corner, "
        "engrossed in picture books about dinosaurs, mermaids, and "
        "rocket ships. The librarian smiled at them on her way past."
    ),
    (
        "He laced up his running shoes and headed out for an "
        "early-morning jog along the river path. The sun had not "
        "yet risen but the sky was already beginning to lighten in "
        "the east. A few other runners passed him in the opposite "
        "direction, nodding silent greetings as they went."
    ),
    (
        "The chef sliced the onions thinly and added them to the "
        "simmering broth. A pot of stock was reducing on the back "
        "burner while a tray of root vegetables roasted in the oven. "
        "The dining room was already filling with the early Tuesday "
        "evening crowd, mostly regulars."
    ),
    (
        "After three weeks of negotiation, the union and the company "
        "finally reached an agreement on wages and overtime rules. "
        "The contract would be sent to the rank-and-file for a vote "
        "the following Monday. Most members expected it to pass, "
        "though a vocal minority wanted to push for more."
    ),
)

# Feature ids — same selection PR #16/#18 used.
FEATURE_IDS: list[int] = [7836, 13953, 15796, 11978]

# The two pairs we report on.
PAIRS: list[tuple[int, int, str]] = [
    (7836, 11978, "within-cluster (highest Polygram overlap, 0.987)"),
    (7836, 15796, "cross-cluster (Polygram MPS 0.464, decoder 0.904)"),
]


def _import_torch_and_transformers():
    try:
        import torch  # noqa: F401
        from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: F401
    except ImportError as exc:
        print(
            f"behavioural_gram_probe: could not import torch + transformers ({exc}); "
            "install them via `pip install torch transformers` and retry.",
            file=sys.stderr,
        )
        return None
    import torch as _torch
    from transformers import GPT2LMHeadModel as _GPT2LMHeadModel
    from transformers import GPT2Tokenizer as _GPT2Tokenizer

    return _torch, _GPT2LMHeadModel, _GPT2Tokenizer


def _load_sae_weights(path: Path) -> dict:
    """Load all four SAE tensors (W_enc, b_enc, W_dec, b_dec) directly,
    bypassing the polygram loader (which only surfaces W_dec)."""
    from safetensors import safe_open

    out: dict = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in ("W_enc", "b_enc", "W_dec", "b_dec"):
            if k not in f.keys():
                raise ValueError(f"SAE checkpoint missing tensor {k!r}")
            out[k] = np.asarray(f.get_tensor(k), dtype=np.float32)
    return out


def _encode(x: np.ndarray, sae: dict) -> np.ndarray:
    """Standard SAELens encoder: f = relu((x - b_dec) @ W_enc + b_enc)."""
    pre = (x - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    return np.maximum(pre, 0.0)


def _polygram_gram_entry(
    records: dict, feature_ids: list[int], i: int, j: int,
) -> float:
    """|<ψ_i|ψ_j>|^2 from Polygram's MPSRung1 encoding for the pair."""
    d, _ = from_sae_lens(records, feature_ids, assign_gamma=True, name="probe")
    g = np.abs(d.gram()) ** 2
    idx_i = feature_ids.index(i)
    idx_j = feature_ids.index(j)
    return float(g[idx_i, idx_j])


def _real_decoder_overlap(sae: dict, i: int, j: int) -> float:
    """Squared cosine of decoder columns i, j."""
    wi = sae["W_dec"][i]
    wj = sae["W_dec"][j]
    num = float(np.dot(wi, wj)) ** 2
    den = float(np.dot(wi, wi)) * float(np.dot(wj, wj))
    return num / den if den > 0 else 0.0


def _co_occurrence(f_i: np.ndarray, f_j: np.ndarray) -> dict:
    """P(j fires | i fires) and P(i fires | j fires) via ReLU>0."""
    fires_i = f_i > 0.0
    fires_j = f_j > 0.0
    n_i = int(fires_i.sum())
    n_j = int(fires_j.sum())
    n_both = int((fires_i & fires_j).sum())
    p_j_given_i = n_both / n_i if n_i else float("nan")
    p_i_given_j = n_both / n_j if n_j else float("nan")
    jaccard = n_both / max(int((fires_i | fires_j).sum()), 1)
    return {
        "n_tokens": int(f_i.shape[0]),
        "n_fires_i": n_i,
        "n_fires_j": n_j,
        "n_fires_both": n_both,
        "p_j_given_i": p_j_given_i,
        "p_i_given_j": p_i_given_j,
        "jaccard": jaccard,
    }


def _activation_pearson(f_i: np.ndarray, f_j: np.ndarray) -> float:
    if f_i.size < 2:
        return float("nan")
    if np.std(f_i) < 1e-12 or np.std(f_j) < 1e-12:
        return float("nan")
    return float(np.corrcoef(f_i, f_j)[0, 1])


def _kl_softmax_row(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Scalar KL(softmax(a) || softmax(b)) for a 1D logit row."""
    log_p = logits_a - float(np.logaddexp.reduce(logits_a, axis=-1))
    log_q = logits_b - float(np.logaddexp.reduce(logits_b, axis=-1))
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


def _run_probe(
    sae_path: Path,
    *,
    n_prompts: int,
    layer: int = 0,
    progress: bool = True,
) -> dict:
    """Returns one report dict containing per-pair behavioural stats
    plus the Polygram + decoder predicted overlaps."""
    deps = _import_torch_and_transformers()
    if deps is None:
        return {"skipped": "torch + transformers not importable"}
    torch, GPT2LMHeadModel, GPT2Tokenizer = deps

    if progress:
        print("Loading GPT-2 small + tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    if progress:
        print("Loading SAE weights...")
    sae = _load_sae_weights(sae_path)

    # Hook the input to model.transformer.h[0]: this is the residual
    # stream entering block 0, equivalent to `blocks.0.hook_resid_pre`
    # in transformer_lens terminology.
    captured: list[np.ndarray] = []

    def _capture_hook(module, args):
        # args[0] is the hidden_states tensor entering the block.
        # Shape: (batch, seq, d_model).
        h = args[0]
        captured.append(h.detach().cpu().numpy())

    handle = model.transformer.h[layer].register_forward_pre_hook(_capture_hook)

    all_residuals: list[np.ndarray] = []
    all_baseline_logits: list[np.ndarray] = []

    if progress:
        print(f"Forwarding {n_prompts} baseline prompts...")
    for k, prompt in enumerate(PROMPTS[:n_prompts]):
        captured.clear()
        toks = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**toks)
        # captured[0]: (1, seq, 768) ; we squeeze batch dim
        all_residuals.append(captured[0][0].astype(np.float32))
        all_baseline_logits.append(
            out.logits[0].detach().cpu().numpy().astype(np.float32)
        )

    handle.remove()

    # Stack across all prompts/tokens.
    residuals = np.concatenate(all_residuals, axis=0)  # (T, 768)
    baseline_logits = np.concatenate(all_baseline_logits, axis=0)  # (T, V)
    n_tokens = residuals.shape[0]
    if progress:
        print(f"Total tokens forwarded: {n_tokens}")

    # Encode the residuals once for all features we care about.
    f_all: dict[int, np.ndarray] = {}
    for fid in FEATURE_IDS:
        # f[i] = relu((x - b_dec) @ W_enc[:, i] + b_enc[i])
        pre = (residuals - sae["b_dec"]) @ sae["W_enc"][:, fid] + sae["b_enc"][fid]
        f_all[fid] = np.maximum(pre, 0.0)

    # Polygram + decoder predictions for all features (we'll select
    # per pair).
    if progress:
        print("Building polygram dictionary for predicted Gram...")
    from polygram import load_sae_safetensors
    records = load_sae_safetensors(sae_path, feature_ids=FEATURE_IDS)
    d_mps, _ = from_sae_lens(records, FEATURE_IDS, assign_gamma=True, name="probe")
    g_polygram = np.abs(d_mps.gram()) ** 2

    # Substitutability: for each pair, run two counterfactual forward
    # passes (one with feature A's contribution subtracted, one with
    # feature B's). We collect per-token KLs at every position where
    # *either* feature fires, so we can both:
    # - report mean KL on tokens where the ablated feature fires
    #   (per-feature impact magnitude)
    # - compute paired substitutability on tokens where *both*
    #   features fire (KL_A vs KL_B on the same tokens).

    def _per_token_ablation_kl(fid: int) -> np.ndarray:
        """Returns an array of shape (n_tokens,) with KL(baseline ||
        ablate-fid) at each token position. Tokens where the feature
        does not fire receive 0.0 (no perturbation, KL = 0)."""
        w_dec_row = sae["W_dec"][fid]  # (768,)
        kl_per_token = np.zeros(n_tokens, dtype=np.float32)
        token_offset = 0
        for prompt in PROMPTS[:n_prompts]:
            toks = tokenizer(prompt, return_tensors="pt")
            seq_len = int(toks["input_ids"].shape[1])
            f_slice = f_all[fid][token_offset : token_offset + seq_len]
            base_slice = baseline_logits[token_offset : token_offset + seq_len]
            fires_idx = np.where(f_slice > 0)[0]
            if fires_idx.size == 0:
                token_offset += seq_len
                continue

            def _ablate_hook(module, args, _fires=fires_idx, _f=f_slice, _w=w_dec_row):
                h = args[0]
                h_np = h.detach().cpu().numpy()[0].copy()
                for t in _fires:
                    h_np[t] = h_np[t] - float(_f[t]) * _w
                new_h = torch.from_numpy(h_np[None, ...]).to(h.dtype).to(h.device)
                return (new_h,) + args[1:]

            handle2 = model.transformer.h[layer].register_forward_pre_hook(_ablate_hook)
            try:
                with torch.no_grad():
                    out = model(**toks)
            finally:
                handle2.remove()
            ablated_logits = (
                out.logits[0].detach().cpu().numpy().astype(np.float32)
            )
            for t in fires_idx:
                # KL is non-negative algebraically; clamp tiny float32
                # noise on near-identical distributions to 0.
                kl_t = max(0.0, _kl_softmax_row(base_slice[t], ablated_logits[t]))
                kl_per_token[token_offset + int(t)] = kl_t
            token_offset += seq_len
        return kl_per_token

    pair_reports: list[dict] = []
    for i, j, label in PAIRS:
        if progress:
            print(f"\n[pair {label}] computing real stats for ({i}, {j})...")
        f_i = f_all[i]
        f_j = f_all[j]
        co = _co_occurrence(f_i, f_j)
        pearson = _activation_pearson(f_i, f_j)
        if progress:
            print(f"  ablation-KL for feature {i} ...")
        kl_i_per = _per_token_ablation_kl(i)
        if progress:
            print(f"  ablation-KL for feature {j} ...")
        kl_j_per = _per_token_ablation_kl(j)

        fires_i = f_i > 0
        fires_j = f_j > 0
        both = fires_i & fires_j
        kl_i_when_i_fires = (
            float(kl_i_per[fires_i].mean()) if fires_i.any() else float("nan")
        )
        kl_j_when_j_fires = (
            float(kl_j_per[fires_j].mean()) if fires_j.any() else float("nan")
        )
        # Paired substitutability — same tokens, two different ablations.
        if both.any():
            kl_i_paired = float(kl_i_per[both].mean())
            kl_j_paired = float(kl_j_per[both].mean())
            kl_paired_n = int(both.sum())
        else:
            kl_i_paired = float("nan")
            kl_j_paired = float("nan")
            kl_paired_n = 0
        idx_i = FEATURE_IDS.index(i)
        idx_j = FEATURE_IDS.index(j)
        pair_reports.append({
            "i": i,
            "j": j,
            "label": label,
            "polygram_overlap": float(g_polygram[idx_i, idx_j]),
            "decoder_overlap": _real_decoder_overlap(sae, i, j),
            "co_occurrence": co,
            "pearson_activation": pearson,
            "kl_ablate_i_when_i_fires": kl_i_when_i_fires,
            "kl_ablate_j_when_j_fires": kl_j_when_j_fires,
            "kl_ablate_i_on_both_fire": kl_i_paired,
            "kl_ablate_j_on_both_fire": kl_j_paired,
            "n_both_fire": kl_paired_n,
        })

    return {
        "layer": layer,
        "n_tokens": n_tokens,
        "n_prompts": n_prompts,
        "pairs": pair_reports,
    }


def _print_report(report: dict) -> None:
    if "skipped" in report:
        print(f"SKIPPED: {report['skipped']}")
        return
    print()
    print("=" * 78)
    print(
        f"BEHAVIOURAL-GRAM PROBE @ blocks.{report['layer']}.hook_resid_pre — "
        f"{report['n_tokens']} tokens across {report['n_prompts']} prompts"
    )
    print("=" * 78)
    for pr in report["pairs"]:
        print()
        print(f"PAIR feat_{pr['i']} ↔ feat_{pr['j']}  ({pr['label']})")
        print("-" * 78)
        print(f"  Polygram predicted overlap:  {pr['polygram_overlap']:.4f}")
        print(f"  Real decoder squared cosine: {pr['decoder_overlap']:.4f}")
        co = pr["co_occurrence"]
        print(
            f"  Co-occurrence (ReLU>0):     "
            f"P(j|i)={co['p_j_given_i']:.4f}  "
            f"P(i|j)={co['p_i_given_j']:.4f}  "
            f"Jaccard={co['jaccard']:.4f}"
        )
        print(
            f"    (i fires on {co['n_fires_i']}/{co['n_tokens']} tokens, "
            f"j fires on {co['n_fires_j']}/{co['n_tokens']}, "
            f"both fire on {co['n_fires_both']})"
        )
        print(f"  Pearson(act_i, act_j):       {pr['pearson_activation']:+.4f}")
        print("  Ablation KL (mean on each feature's fire tokens):")
        print(
            f"    KL(baseline || ablate-i) when i fires: "
            f"{pr['kl_ablate_i_when_i_fires']:.3e}"
        )
        print(
            f"    KL(baseline || ablate-j) when j fires: "
            f"{pr['kl_ablate_j_when_j_fires']:.3e}"
        )
        if pr["n_both_fire"] > 0:
            print(
                f"  Paired substitutability on {pr['n_both_fire']} "
                "tokens where BOTH features fire:"
            )
            print(
                f"    KL(baseline || ablate-i) on both-fire tokens: "
                f"{pr['kl_ablate_i_on_both_fire']:.3e}"
            )
            print(
                f"    KL(baseline || ablate-j) on both-fire tokens: "
                f"{pr['kl_ablate_j_on_both_fire']:.3e}"
            )
            if (
                pr["kl_ablate_i_on_both_fire"] > 0
                and pr["kl_ablate_j_on_both_fire"] > 0
            ):
                ratio = (
                    pr["kl_ablate_i_on_both_fire"]
                    / pr["kl_ablate_j_on_both_fire"]
                )
                print(
                    f"    KL ratio i/j on both-fire tokens = {ratio:.3f} "
                    "(substitutable when in [0.5, 2.0])"
                )
        else:
            print("  Paired substitutability: skipped (no both-fire tokens)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=len(PROMPTS),
        help=f"how many prompts to forward (1..{len(PROMPTS)})",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress prints"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=0,
        choices=(0, 5, 10),
        help=(
            "Which GPT-2 block input to hook + which SAE checkpoint to "
            "load. The same `jbloom/GPT2-Small-SAEs-Reformatted` repo "
            "ships SAEs for blocks {0, 5, 10}.hook_resid_pre. Note that "
            "FEATURE_IDS were chosen on the layer-0 SAE; at layers 5 / "
            "10 the same indices reference *different* SAE features. "
            "The probe still measures meaningful per-feature ablation-KL "
            "magnitudes — that is the §4.3 question — but per-pair "
            "co-occurrence and substitutability metrics across layers "
            "are not comparing the same semantic feature."
        ),
    )
    args = parser.parse_args(argv)

    sae_path = Path(
        f"./scratch/real-sae/blocks.{args.layer}.hook_resid_pre/"
        "sae_weights.safetensors"
    )
    if not sae_path.exists():
        print(
            f"behavioural_gram_probe: SAE checkpoint not found at "
            f"{sae_path}. Download with `hf download "
            f"jbloom/GPT2-Small-SAEs-Reformatted "
            f"--include='blocks.{args.layer}.hook_resid_pre/sae_weights.safetensors' "
            f"--local-dir ./scratch/real-sae`. Skipping.",
            file=sys.stderr,
        )
        return

    n_prompts = max(1, min(args.n_prompts, len(PROMPTS)))
    report = _run_probe(
        sae_path, n_prompts=n_prompts, layer=args.layer, progress=not args.quiet
    )
    _print_report(report)


# Suppress unused-import warning: MPSRung1 stays exported in case
# follow-ups want to swap encodings without re-importing.
_ = (Dictionary, MPSRung1)


if __name__ == "__main__":
    main(sys.argv[1:])
