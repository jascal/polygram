"""Behavioural-Gram scale-up — Polygram → behavioural-Jaccard
correlation across 28 pairs at `blocks.10.hook_resid_pre` on GPT-2
small.

Per `tech-debt-backlog/tasks.md` §4.4. PR #20 (§4.2) settled the
*directional* claim on N = 2 pairs: Polygram's within-vs-cross ordering
is preserved at the behavioural level, but per-pair Jaccard magnitudes
compress vs Polygram-predicted overlaps. PR #23 (§4.3) settled the
layer choice: hook at `blocks.5+` so that ablation-KL is informative.
What's still open is the *shape* of the Polygram → behavioural
correspondence across many pairs at a usable layer.

Scope adjustment vs the merged spec
-----------------------------------
The §4.4 spec named "~25 features → ~300 pairs", picked stratified
across the predicted-overlap distribution. Polygram's rung-1 MPS
encoding caps a Dictionary at 8 features
(`MAX_FEATURES_PER_DICTIONARY` in `polygram/sae_import.py`). This
script implements the cap-respecting variant: 8 features → 28 pairs,
selected at `blocks.10.hook_resid_pre` via projection-similarity to a
high-firing seed, with stratification across the resulting cosine
distribution. Still ~4.7× pair count vs PR #18's N = 6 and meets the
spec's "30+ pairs" floor approximately. The slope estimate over 28
pairs is statistically usable; the spec's three-bucket reporting
becomes coarser (~9 pairs per bucket) but still informative.

Pipeline
--------
1. Load `blocks.10.hook_resid_pre` SAE; pick a high-firing seed
   feature, then 7 features stratified by decoder cosine to that seed
   (4 high-cosine "near-cluster", 4 low-cosine "far-cluster").
2. Build a Polygram Dictionary via `from_sae_lens` (KMeans k = 2 on
   the 8 projection vectors, `assign_gamma=True` for γ spread).
3. Forward the §4.2 / §4.3 12-prompt set through GPT-2 small,
   capture residuals at `blocks.10`, encode through the SAE → 8
   activation vectors of length ~654.
4. Run 8 ablation forward passes, one per feature, computing
   per-token KL on the next-token distribution wherever that feature
   fires.
5. Compute per-pair behavioural stats (Jaccard, activation Pearson,
   paired ablation-KL ratio on both-fire tokens with ≥ 5 both-fire
   tokens).
6. Report Spearman + Pearson between Polygram and each behavioural
   metric; same for decoder cosine (ceiling); per-bucket Jaccard
   means with 95% bootstrap CI.
7. Emit `docs/research/data/scaleup_pairs.csv` with the per-pair row
   set for re-analysis.

Reproducibility
---------------

    python examples/behavioural_gram_scaleup.py

Skip path: SAE checkpoint absent → exits with hint pointing at the
`hf download jbloom/GPT2-Small-SAEs-Reformatted` command. `torch` /
`transformers` import failures handled gracefully with a hint.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from polygram import Dictionary, MPSRung1
from polygram.sae_import import from_sae_lens, load_sae_safetensors

# Same prompt set as §4.2 / §4.3 — keeps token statistics comparable.
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

# Cap from polygram/sae_import.py:23 — rung-1 MPS encoding limit.
MAX_FEATURES = 8


def _import_torch_and_transformers():
    try:
        import torch  # noqa: F401
        from transformers import GPT2LMHeadModel, GPT2Tokenizer  # noqa: F401
    except ImportError as exc:
        print(
            f"behavioural_gram_scaleup: could not import torch + transformers ({exc}); "
            "install them via `pip install torch transformers` and retry.",
            file=sys.stderr,
        )
        return None
    import torch as _torch
    from transformers import GPT2LMHeadModel as _GPT2LMHeadModel
    from transformers import GPT2Tokenizer as _GPT2Tokenizer

    return _torch, _GPT2LMHeadModel, _GPT2Tokenizer


def _load_sae_weights(path: Path) -> dict:
    from safetensors import safe_open

    out: dict = {}
    with safe_open(str(path), framework="numpy") as f:
        for k in ("W_enc", "b_enc", "W_dec", "b_dec"):
            if k not in f.keys():
                raise ValueError(f"SAE checkpoint missing tensor {k!r}")
            out[k] = np.asarray(f.get_tensor(k), dtype=np.float32)
    return out


def _kl_softmax_row(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    log_p = logits_a - float(np.logaddexp.reduce(logits_a, axis=-1))
    log_q = logits_b - float(np.logaddexp.reduce(logits_b, axis=-1))
    p = np.exp(log_p)
    return float(np.sum(p * (log_p - log_q)))


def _select_features(
    sae: dict,
    *,
    n_features: int,
    seed_candidates: list[int],
    firing_rates: np.ndarray,
    min_firing_rate: float,
    progress: bool,
) -> list[int]:
    """Pick `n_features` features stratified by decoder cosine to a
    high-firing seed feature.

    Strategy:
    - Among `seed_candidates`, pick the seed that maximizes firing
      rate * decoder-norm (high-impact feature). Acts as the
      "anchor" of the near-cluster.
    - Compute cosine of every feature's decoder column to the seed.
    - Filter to features with firing rate ≥ `min_firing_rate` (so
      that Jaccard is meaningful — features that fire on < 1% of
      tokens give Jaccard ≈ 0 regardless of pair structure).
    - Take the top-`n_features // 2` by cosine (excluding seed) → the
      "near" subset, plus `n_features // 2 - 1` features stratified
      across the rest of the cosine distribution → the "far" subset.
      Total = `n_features` (seed + near + far).

    The result is fed directly into `from_sae_lens` — KMeans with
    k = 2 on these 8 projection vectors will, by construction, recover
    the near/far split (the cosine cutoff *is* a cluster split in
    decoder-space).
    """
    w_dec = sae["W_dec"]  # (n_features_total, d_model)
    norms = np.linalg.norm(w_dec, axis=1)
    n_total = w_dec.shape[0]

    # Candidate seeds: those with firing rate ≥ min_firing_rate AND in
    # the user-specified candidate pool (default: full SAE, but
    # constrained to firing).
    eligible = np.where(firing_rates >= min_firing_rate)[0]
    if eligible.size < n_features:
        raise RuntimeError(
            f"only {eligible.size} features fire at rate ≥ "
            f"{min_firing_rate}; need at least {n_features}. "
            f"Try lowering --min-firing-rate."
        )

    if seed_candidates:
        eligible_set = set(eligible.tolist())
        seed_pool = [s for s in seed_candidates if s in eligible_set]
        if not seed_pool:
            raise RuntimeError(
                f"no seed_candidates fire at rate ≥ {min_firing_rate}; "
                f"either lower the threshold or pick different candidates."
            )
    else:
        seed_pool = eligible.tolist()

    # Pick the seed: highest firing-rate × decoder-norm among the pool.
    impact = firing_rates[seed_pool] * norms[seed_pool]
    seed = int(seed_pool[int(np.argmax(impact))])
    if progress:
        print(
            f"  seed feature = {seed} "
            f"(firing rate {firing_rates[seed]:.3f}, "
            f"decoder norm {norms[seed]:.3f})"
        )

    # Cosine of seed against every other eligible feature.
    seed_vec = w_dec[seed]
    seed_unit = seed_vec / max(np.linalg.norm(seed_vec), 1e-12)
    eligible_other = np.array([i for i in eligible if i != seed])
    others = w_dec[eligible_other]
    others_unit = others / np.maximum(
        np.linalg.norm(others, axis=1, keepdims=True), 1e-12
    )
    cos = others_unit @ seed_unit  # (n_eligible-1,)

    n_near = (n_features - 1) // 2  # 3 if n_features = 8
    n_far = (n_features - 1) - n_near  # 4 if n_features = 8 → seed+3+4=8

    # Near: top-n_near by cosine.
    order = np.argsort(-cos)
    near = eligible_other[order[:n_near]].tolist()

    # Far: stratify across the bottom half of the cosine distribution.
    bottom_half = order[len(order) // 2 :]
    if bottom_half.size < n_far:
        raise RuntimeError(
            f"not enough far-cosine eligible features "
            f"({bottom_half.size}) for n_far={n_far}"
        )
    step = max(1, bottom_half.size // n_far)
    far = eligible_other[bottom_half[::step][:n_far]].tolist()

    selected = [seed] + near + far
    if len(selected) != n_features:
        raise RuntimeError(
            f"selection produced {len(selected)} features, expected {n_features}"
        )

    if progress:
        print("  selected features (id, firing_rate, cosine_to_seed):")
        for fid in selected:
            if fid == seed:
                cos_to_seed = 1.0
            else:
                idx_in_other = int(
                    np.where(eligible_other == fid)[0][0]
                )
                cos_to_seed = float(cos[idx_in_other])
            print(
                f"    feat_{fid:5d}  rate={firing_rates[fid]:.3f}  "
                f"cos_to_seed={cos_to_seed:+.3f}"
            )
    return selected


def _bootstrap_ci_mean(
    values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = np.array(
        [rng.choice(values, size=values.size, replace=True).mean()
         for _ in range(n_resamples)]
    )
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _run_probe(
    sae_path: Path,
    *,
    n_prompts: int,
    n_features: int,
    seed_candidates: list[int],
    min_firing_rate: float,
    progress: bool,
) -> dict:
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
        print("Loading SAE weights (blocks.10)...")
    sae = _load_sae_weights(sae_path)

    layer = 10
    captured: list[np.ndarray] = []

    def _capture_hook(module, args):
        captured.append(args[0].detach().cpu().numpy())

    handle = model.transformer.h[layer].register_forward_pre_hook(_capture_hook)

    all_residuals: list[np.ndarray] = []
    all_baseline_logits: list[np.ndarray] = []
    if progress:
        print(f"Forwarding {n_prompts} baseline prompts...")
    for prompt in PROMPTS[:n_prompts]:
        captured.clear()
        toks = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model(**toks)
        all_residuals.append(captured[0][0].astype(np.float32))
        all_baseline_logits.append(
            out.logits[0].detach().cpu().numpy().astype(np.float32)
        )
    handle.remove()

    residuals = np.concatenate(all_residuals, axis=0)
    baseline_logits = np.concatenate(all_baseline_logits, axis=0)
    n_tokens = residuals.shape[0]
    if progress:
        print(f"Total tokens forwarded: {n_tokens}")

    # Encode the entire SAE in a single pass on the residuals so we
    # can compute firing rates for selection.
    if progress:
        print("Encoding all features for firing-rate selection...")
    pre = (residuals - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    activations_all = np.maximum(pre, 0.0)  # (n_tokens, n_features_total)
    firing_rates = (activations_all > 0.0).mean(axis=0)
    if progress:
        n_active = int((firing_rates >= min_firing_rate).sum())
        print(
            f"  {n_active} features fire on ≥ {min_firing_rate} of "
            f"{n_tokens} tokens"
        )

    if progress:
        print(f"Selecting {n_features} features stratified by decoder cosine...")
    selected_ids = _select_features(
        sae,
        n_features=n_features,
        seed_candidates=seed_candidates,
        firing_rates=firing_rates,
        min_firing_rate=min_firing_rate,
        progress=progress,
    )

    # Build polygram dictionary from the selected ids.
    if progress:
        print("Building Polygram Dictionary via from_sae_lens...")
    records = load_sae_safetensors(sae_path, feature_ids=selected_ids)
    d_mps, report = from_sae_lens(
        records, selected_ids, assign_gamma=True, name="ScaleupBlocks10",
    )
    g_polygram = np.abs(d_mps.gram()) ** 2

    # Cache per-feature activations for the selected ids.
    f_per_feature = {
        fid: activations_all[:, fid].astype(np.float32) for fid in selected_ids
    }

    # Run ablation passes — one per selected feature.
    def _per_token_ablation_kl(fid: int) -> np.ndarray:
        w_dec_row = sae["W_dec"][fid]
        kl_per_token = np.zeros(n_tokens, dtype=np.float32)
        token_offset = 0
        for prompt in PROMPTS[:n_prompts]:
            toks = tokenizer(prompt, return_tensors="pt")
            seq_len = int(toks["input_ids"].shape[1])
            f_slice = f_per_feature[fid][token_offset : token_offset + seq_len]
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
                kl_t = max(
                    0.0, _kl_softmax_row(base_slice[t], ablated_logits[t])
                )
                kl_per_token[token_offset + int(t)] = kl_t
            token_offset += seq_len
        return kl_per_token

    if progress:
        print(f"Running {len(selected_ids)} ablation passes...")
    kl_per_feature: dict[int, np.ndarray] = {}
    for k, fid in enumerate(selected_ids):
        if progress:
            print(f"  [{k+1}/{len(selected_ids)}] ablation-KL for feat_{fid}")
        kl_per_feature[fid] = _per_token_ablation_kl(fid)

    # Per-pair stats.
    pair_rows: list[dict] = []
    for i_idx, i in enumerate(selected_ids):
        for j_idx, j in enumerate(selected_ids):
            if j_idx <= i_idx:
                continue
            f_i = f_per_feature[i]
            f_j = f_per_feature[j]
            fires_i = f_i > 0.0
            fires_j = f_j > 0.0
            both = fires_i & fires_j
            either = fires_i | fires_j
            n_both = int(both.sum())
            n_either = int(either.sum())
            jaccard = n_both / max(n_either, 1)

            if np.std(f_i) > 1e-12 and np.std(f_j) > 1e-12 and f_i.size >= 2:
                pearson = float(np.corrcoef(f_i, f_j)[0, 1])
            else:
                pearson = float("nan")

            # Decoder cosine².
            wi = sae["W_dec"][i]
            wj = sae["W_dec"][j]
            denom = float(np.dot(wi, wi)) * float(np.dot(wj, wj))
            decoder_overlap = (
                float(np.dot(wi, wj)) ** 2 / denom if denom > 0 else 0.0
            )

            polygram_overlap = float(g_polygram[i_idx, j_idx])

            kl_i = kl_per_feature[i]
            kl_j = kl_per_feature[j]
            if n_both >= 5:
                kl_i_paired = float(kl_i[both].mean())
                kl_j_paired = float(kl_j[both].mean())
                if kl_i_paired > 0 and kl_j_paired > 0:
                    kl_ratio = kl_i_paired / kl_j_paired
                    kl_dist_from_1 = abs(np.log(kl_ratio))
                else:
                    kl_ratio = float("nan")
                    kl_dist_from_1 = float("nan")
            else:
                kl_i_paired = float("nan")
                kl_j_paired = float("nan")
                kl_ratio = float("nan")
                kl_dist_from_1 = float("nan")

            pair_rows.append({
                "i": i,
                "j": j,
                "polygram_overlap": polygram_overlap,
                "decoder_overlap": decoder_overlap,
                "jaccard": jaccard,
                "pearson_activation": pearson,
                "n_fires_i": int(fires_i.sum()),
                "n_fires_j": int(fires_j.sum()),
                "n_both_fire": n_both,
                "n_either_fire": n_either,
                "kl_ablate_i_on_both_fire": kl_i_paired,
                "kl_ablate_j_on_both_fire": kl_j_paired,
                "kl_ratio_i_over_j": kl_ratio,
                "kl_log_ratio_abs": kl_dist_from_1,
            })

    return {
        "layer": layer,
        "n_tokens": n_tokens,
        "n_prompts": n_prompts,
        "n_features": len(selected_ids),
        "selected_feature_ids": selected_ids,
        "cluster_method": report.cluster_method,
        "beta_variance_explained": report.beta_variance_explained,
        "pairs": pair_rows,
    }


def _summarize(report: dict) -> dict:
    """Compute Spearman/Pearson and per-bucket Jaccard means."""
    pairs = report["pairs"]
    polygram = np.array([p["polygram_overlap"] for p in pairs])
    decoder = np.array([p["decoder_overlap"] for p in pairs])
    jaccard = np.array([p["jaccard"] for p in pairs])
    pearson_act = np.array(
        [p["pearson_activation"] for p in pairs], dtype=float
    )
    log_kl = np.array([p["kl_log_ratio_abs"] for p in pairs], dtype=float)

    out = {
        "n_pairs": len(pairs),
        "spearman": {
            "polygram_jaccard": _spearman(polygram, jaccard),
            "decoder_jaccard": _spearman(decoder, jaccard),
            "polygram_pearson_activation": _spearman(
                polygram, np.where(np.isnan(pearson_act), 0.0, pearson_act)
            ),
        },
        "pearson": {
            "polygram_jaccard": _pearson(polygram, jaccard),
            "decoder_jaccard": _pearson(decoder, jaccard),
        },
    }

    valid_kl = ~np.isnan(log_kl)
    if valid_kl.sum() >= 2:
        out["spearman"]["polygram_log_kl_abs"] = _spearman(
            polygram[valid_kl], log_kl[valid_kl]
        )
        out["n_pairs_with_kl"] = int(valid_kl.sum())
    else:
        out["spearman"]["polygram_log_kl_abs"] = float("nan")
        out["n_pairs_with_kl"] = int(valid_kl.sum())

    # Per-bucket Jaccard means.
    low = jaccard[polygram <= 0.4]
    mid = jaccard[(polygram > 0.4) & (polygram < 0.7)]
    hi = jaccard[polygram >= 0.7]
    out["buckets"] = {
        "low_overlap": {
            "polygram_range": "≤ 0.4",
            "n_pairs": int(low.size),
            "jaccard_mean": float(low.mean()) if low.size else float("nan"),
            "jaccard_ci_95": _bootstrap_ci_mean(low),
        },
        "mid_overlap": {
            "polygram_range": "(0.4, 0.7)",
            "n_pairs": int(mid.size),
            "jaccard_mean": float(mid.mean()) if mid.size else float("nan"),
            "jaccard_ci_95": _bootstrap_ci_mean(mid),
        },
        "high_overlap": {
            "polygram_range": "≥ 0.7",
            "n_pairs": int(hi.size),
            "jaccard_mean": float(hi.mean()) if hi.size else float("nan"),
            "jaccard_ci_95": _bootstrap_ci_mean(hi),
        },
    }

    # Outcome bucket per spec.
    sp = out["spearman"]["polygram_jaccard"]
    if not np.isnan(sp):
        if sp >= 0.6:
            out["outcome"] = "high_spearman_loop_unblocked"
        elif sp >= 0.3:
            out["outcome"] = "medium_spearman_loop_needs_calibration"
        else:
            out["outcome"] = "low_spearman_loop_blocked"
    else:
        out["outcome"] = "undefined"
    return out


def _print_report(report: dict, summary: dict) -> None:
    if "skipped" in report:
        print(f"SKIPPED: {report['skipped']}")
        return
    print()
    print("=" * 78)
    print(
        f"BEHAVIOURAL-GRAM SCALE-UP @ blocks.{report['layer']}.hook_resid_pre — "
        f"{report['n_features']} features, {summary['n_pairs']} pairs, "
        f"{report['n_tokens']} tokens"
    )
    print("=" * 78)
    print(
        f"Cluster method: {report['cluster_method']} "
        f"(β var-explained {report['beta_variance_explained']:.3f})"
    )
    print()
    print("Correlations (Polygram + decoder vs behavioural metrics):")
    sp = summary["spearman"]
    pe = summary["pearson"]
    print(f"  Spearman(Polygram, Jaccard):           {sp['polygram_jaccard']:+.4f}")
    print(f"  Spearman(decoder,  Jaccard):           {sp['decoder_jaccard']:+.4f}")
    print(
        f"  Spearman(Polygram, log|KL_i/KL_j|):    "
        f"{sp['polygram_log_kl_abs']:+.4f}  "
        f"(n_pairs_with_kl={summary['n_pairs_with_kl']})"
    )
    print(f"  Pearson(Polygram,  Jaccard):           {pe['polygram_jaccard']:+.4f}")
    print(f"  Pearson(decoder,   Jaccard):           {pe['decoder_jaccard']:+.4f}")
    print()
    print("Per-bucket Jaccard means (95% bootstrap CI):")
    for name, b in summary["buckets"].items():
        ci = b["jaccard_ci_95"]
        print(
            f"  {name:14s} (Polygram {b['polygram_range']:9s}, "
            f"n={b['n_pairs']:2d}): "
            f"Jaccard={b['jaccard_mean']:.4f}  CI=[{ci[0]:.4f}, {ci[1]:.4f}]"
        )
    print()
    print(f"OUTCOME: {summary['outcome']}")


def _write_csv(report: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = report["pairs"]
    if not rows:
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=len(PROMPTS),
        help=f"how many prompts to forward (1..{len(PROMPTS)})",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=MAX_FEATURES,
        help=(
            f"how many features to select for the scaleup probe "
            f"(2..{MAX_FEATURES}, capped at MAX_FEATURES_PER_DICTIONARY)"
        ),
    )
    parser.add_argument(
        "--min-firing-rate",
        type=float,
        default=0.01,
        help=(
            "minimum firing rate (fraction of tokens where the SAE "
            "feature has post-ReLU activation > 0) for a feature to be "
            "eligible. Filters out near-dead features whose Jaccard "
            "would be 0 regardless of pair structure."
        ),
    )
    parser.add_argument(
        "--seed-candidates",
        type=int,
        nargs="*",
        default=[],
        help=(
            "optional pool of feature ids to consider as the selection "
            "anchor (the seed). If empty, the script picks the highest-"
            "firing eligible feature globally. Useful for reproducibility "
            "across runs."
        ),
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("docs/research/data/scaleup_pairs.csv"),
        help="path to write the per-pair CSV (skipped on dry-skip path).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress progress prints"
    )
    args = parser.parse_args(argv)

    if not (2 <= args.n_features <= MAX_FEATURES):
        parser.error(
            f"--n-features must be in [2, {MAX_FEATURES}]; "
            f"got {args.n_features}"
        )

    sae_path = Path(
        "./scratch/real-sae/blocks.10.hook_resid_pre/sae_weights.safetensors"
    )
    if not sae_path.exists():
        print(
            f"behavioural_gram_scaleup: SAE checkpoint not found at "
            f"{sae_path}. Download with `hf download "
            f"jbloom/GPT2-Small-SAEs-Reformatted "
            f"--include='blocks.10.hook_resid_pre/sae_weights.safetensors' "
            f"--local-dir ./scratch/real-sae`. Skipping.",
            file=sys.stderr,
        )
        return

    n_prompts = max(1, min(args.n_prompts, len(PROMPTS)))
    report = _run_probe(
        sae_path,
        n_prompts=n_prompts,
        n_features=args.n_features,
        seed_candidates=list(args.seed_candidates),
        min_firing_rate=args.min_firing_rate,
        progress=not args.quiet,
    )
    if "skipped" in report:
        _print_report(report, {})
        return
    summary = _summarize(report)
    _print_report(report, summary)
    _write_csv(report, args.csv_out)
    if not args.quiet:
        print(f"\nPer-pair CSV written to {args.csv_out}")


# Suppress unused-import warning: MPSRung1, Dictionary stay exported
# in case follow-ups want to swap encodings.
_ = (Dictionary, MPSRung1)


if __name__ == "__main__":
    main(sys.argv[1:])
