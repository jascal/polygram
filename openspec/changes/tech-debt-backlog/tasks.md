# tech-debt-backlog — tasks

Rolling list of small items not worth their own OpenSpec change.
Group by area. Mark `[x]` when shipped; leave the task in-place
with a one-line note describing what was done and any commit/PR
reference.

## 1. Documentation

- [ ] 1.1 README — add a short "Choosing an encoding" section that
      contrasts `MPSRung1` (analytic structural floor; rung-1
      staircase; cheap to verify) with `HEA_Rung2` (richer θ
      tensor; depth/entangler/rotation knobs; tier-separation
      invariant available; no closed-form floor in the multi-knob
      case). Two short paragraphs + a one-line "use MPS by default,
      reach for HEA when you need expressivity beyond a single
      Pauli-Rz phase axis" recommendation.
      (Source: code review on PR #4, suggestion 1.)

## 2. Cancellation diagnostics — research-track follow-ups

- [~] 2.1 Per-knob HEA floor diagnostics (research-track). Add
      `result.structural_floor(knob="featureX.theta[r,d,q]")` (1D
      scan minimum, holding other knobs fixed) and
      `structural_floor_joint` (numerical lower bound from a short
      optimization run). Both are *numerical bounds*, not analytic
      structural floors — `extend-cancellation-sweep-hea` deliberately
      rejected the "structural" label for best-found-so-far values.
      Empirical evidence from the same change strengthens the case:
      4-θ Ry knobs drive `(dog_poodle, bird_hawk)` overlap to ≈ 0
      *while shattering cluster invariants* (sibling overlap
      0.9999 → 0.58, tier-separation +0.22 → −0.20). A per-knob
      "floor" reads as a guarantee but cross-knob interactions can
      punch through it. The right shape is probably a
      **cluster-respecting** knob set (shared θ across siblings, or
      invariant-preserving optimization with `tier_separation_bound`
      as a hard constraint) — see the `Out of Scope` bullet in the
      archived proposal at
      `openspec/changes/archive/2026-05-03-extend-cancellation-sweep-hea/proposal.md`.
      Defer until an SAE workload tells us what shape of bound is
      useful. (Source: dev-agent task spec received 2026-05-03;
      deferred per ad-hoc decision after empirical θ-experiment
      surfaced cross-term cluster hazard.)
      *Superseded 2026-05-03 by `add-cluster-shared-knobs`* — the
      principled answer is a binding mechanic (cluster-shared knobs
      that collapse per-feature axes to one-per-cluster), not a
      per-knob "floor" diagnostic. Bit-for-bit Gram preservation
      holds for `MPSRung1 <cluster>.phi`; HEA cluster-shared paths
      ship as a search-space dimensionality reduction (no algebraic
      bound). Per-knob/joint floor diagnostics remain deferred.

## 3. Encoding-invariance verification — research-track follow-up

- [x] 3.1 Encoding-invariance spike (MPS vs HEA classification
      stability). Both `add-sharing-graph-triage` and
      `add-batch-experiment` ride on the rung-1 closed-form
      `(M, V, structural_floor, cancellation_gap)` decomposition,
      which is exact for `MPSRung1` and only-approximate for
      `HEA_Rung2` (the analytic floor isn't defined in the multi-knob
      HEA case — see the `structural_floor()`
      `NotImplementedError` rail in `polygram/cancellation.py`).
      The open question is whether a feature pair classified as
      "good sharing candidate" or "must separate" under `MPSRung1`
      stays in that bucket when re-encoded as `HEA_Rung2(depth=2)`
      (or vice-versa). If classifications drift across encodings,
      the `BatchResults` carrying both predictions and observations
      will silently disagree with the input `FeatureGraph` on
      genuine SAE workloads.
      Concrete plan: pick one fixture (the `tests/fixtures/toy_sae.json`
      4-feature subset is enough), run `triage_dictionary` →
      `build_separation_graph` and `build_sharing_graph` against
      both `MPSRung1` and `HEA_Rung2(depth=2)` instantiations of the
      same `(name, hierarchy, beta, alpha, gamma, phi)` features,
      and compare:
      (a) the kept-edge set per kind;
      (b) per-pair `(M, V, structural_floor)` distances;
      (c) `BatchExperiment(top_k=4).run().runs` outputs.
      Ship as a research note under `docs/research/` once empirical
      data exists. This blocks any future
      compression-pipeline work that depends on the predictions
      being stable across encodings.
      (Source: `add-batch-experiment` proposal Out of Scope; flagged
      while landing the `BatchExperiment` consumer of `FeatureGraph`.)

      *Closed 2026-05-04*. Empirical findings landed in
      `docs/research/cross-encoding-stability.md` and the
      reproducible script `examples/cross_encoding_stability.py`. TL;DR
      across three fixtures (Animals, toy SAE, real GPT-2 SAE):
      kept-edge classifications agree perfectly between MPSRung1
      and HEA_Rung2(depth=2) at default thresholds; per-pair
      magnitudes differ systematically on cross-cluster pairs (HEA
      reports up to +0.30 higher current_overlap and collapses pairs
      sharing |Δβ| toward each other). Practical recommendation:
      use rung-1 triage as a cheap pre-filter for pair selection
      (its classifications transfer); regenerate magnitudes in the
      target encoding before quantitative conclusions. Follow-up
      worth running: depth-vs-γ-leverage at HEA depth=4/8 to confirm
      depth=2 is the cause of cross-cluster magnitude collapse.

## 4. Encoding validity vs ground truth — research-track follow-up

- [x] 4.1 Decoder-Gram validity spike. PR #16
      (`cross-encoding-stability`) closed the *internal* consistency
      question: MPS and HEA agree on which pairs cross the kept-edge
      gates. It explicitly leaves open the *external* validity
      question — does either encoding's predicted Gram track the
      actual SAE decoder geometry it claims to encode? The note's
      own closing caveat names this gap: "compares two encodings to
      each other, not either encoding to actual SAE behaviour on
      text." Until this is answered, every downstream prediction —
      `BatchExperiment.cancellation_efficiency`, the
      `build_separation_graph` "must-separate" flagging, the entire
      disentanglement-loop sketch deferred in
      `docs/research/spec-disentanglement-loop.md` — could be
      pointing at a signal that lives only inside the encoded
      representation, not the SAE.
      The right test is the smallest one that can falsify the
      assumption. Operationally:
      (a) pick the same real GPT-2 SAE feature subset PR #16 used
      (`feat_7836`, `feat_13953`, `feat_15796`, `feat_11978` from
      `jbloom/GPT2-Small-SAEs-Reformatted`'s
      `blocks.0.hook_resid_pre`);
      (b) compute the *raw* decoder squared-cosine Gram
      `G_real[i,j] = (W_dec[:,i] · W_dec[:,j])² /
      (‖W_dec[:,i]‖² · ‖W_dec[:,j]‖²)` directly from the SAE
      safetensors — no polygram involved;
      (c) run `from_sae_lens` to build a `Dictionary` and compute
      `G_polygram[i,j] = |⟨ψ_i|ψ_j⟩|²` via `Dictionary.gram()`
      under both `MPSRung1` and `HEA_Rung2(depth=2)`;
      (d) compare: per-pair scatter, Spearman rank correlation,
      max absolute drift; report whether classifications (sharing /
      separation / floor-block) computed from `G_real` agree with
      those computed from `G_polygram`.
      Three outcomes shape the next move:
      - Spearman > 0.8 across both encodings: encoding tracks real
        geometry. The first of the three blockers in
        `spec-disentanglement-loop.md` (gradient signal exists)
        gets meaningful evidence.
      - Spearman 0.3–0.8: encoding tracks real geometry on average
        but loses fine structure. Polygram is a useful *ranker*;
        quantitative claims need per-workload calibration.
      - Spearman < 0.3: encoding reads geometry the SAE doesn't
        have. `from_sae_lens`'s lossy projection (PCA per cluster
        + KMeans cluster assignment) discards the load-bearing
        signal. Disentanglement loop blocked indefinitely;
        `from_sae_lens` itself needs rethinking before any
        compression-pipeline work proceeds.
      Ship as `docs/research/decoder-gram-validity.md` plus the
      reproducible `examples/decoder_gram_validity.py` script,
      same shape as the cross-encoding spike. No new polygram
      surface required — the script reads safetensors, calls
      `from_sae_lens`, calls `Dictionary.gram()`, computes
      correlations.
      Blocks: any compression-pipeline or real-model-validation
      work (including the user-supplied "Spec-DisEntanglement Loop
      v0.1" sketch). The Gemma-Scope intervention pipeline that
      sketch describes is several layers downstream of this
      question; if the answer here is "no correlation," the entire
      Gemma harness investigates a phantom.
      (Source: 2026-05-04 sketch from a peer agent proposing a
      full Gemma-Scope steering / reconstruction validation loop;
      flagged that the load-bearing assumption — encoded
      interference reflects real decoder geometry — has never been
      tested directly. PR #16 confirmed encoding-invariance but
      explicitly left external validity open.)

      *Closed 2026-05-04*. Empirical findings landed in
      `docs/research/decoder-gram-validity.md` and the reproducible
      script `examples/decoder_gram_validity.py`. TL;DR across two
      fixtures (Toy SAE, Real GPT-2 SAE): on the Real SAE both
      encodings hit Spearman 0.94 vs the real decoder squared-cosine
      Gram (top outcome bucket — encoding tracks real ranking); on
      Toy SAE the same encodings land in the middle bucket
      (Spearman 0.54 MPS, 0.66 HEA). Per-pair magnitudes diverge by
      up to 0.44 squared-overlap units even on the Real SAE
      (Pearson +0.74–+0.90). Structural reason: Polygram's
      `(β, α, γ, φ)` parameterization with `β ∈ [-0.5, 0.5]` puts a
      floor on cross-cluster squared overlap (≈ 0.4 MPS, ≈ 0.73
      HEA depth=2); real decoders aren't bound by that floor, so
      whenever an SAE has cleanly orthogonal cross-cluster geometry
      Polygram over-predicts. Practical recommendation: treat
      Polygram as a *ranker* not a magnitude predictor. The first
      blocker in `spec-disentanglement-loop.md` ("gradient signal
      exists") gets partial evidence — ranking signal is real, so
      a primitive that uses Polygram to *order* candidate pairs
      operates on real information; one that uses Polygram
      magnitudes as a quantitative loss surface would be optimizing
      encoding-internal artefacts. Downstream work that rides on
      ranking (BatchExperiment top-K selection, sharing/separation
      kept-edge sets) is unblocked; work that depends on magnitudes
      (the disentanglement-loop loss surface; the user-supplied
      Gemma `compression_score` formula) needs a different signal
      source for the magnitude inputs.

- [x] 4.2 Behavioural-Gram one-pair probe (research-track). PR #18
      settled the *decoder-geometry* validity question
      (`G_polygram` vs decoder squared-cosine Gram). Its closing
      caveat names the next gap explicitly: "Two SAE features can
      have orthogonal decoder columns but still co-fire on the same
      inputs (and vice versa). The behavioural-Gram comparison
      would need a forward-pass infrastructure Polygram doesn't
      have." This task is the smallest probe that builds that
      infrastructure for a single pair and tests whether Polygram's
      ranking signal carries into real-model behaviour.
      Scope is deliberately narrow:
      - **One model.** GPT-2 small + the
        `jbloom/GPT2-Small-SAEs-Reformatted` `blocks.0.hook_resid_pre`
        SAE that PR #16 and PR #18 already use. No Gemma, no
        multi-layer, no SAE-format generalization — that's its own
        future task.
      - **One pair.** One within-cluster pair from PR #18's
        feature subset (`feat_7836, feat_13953, feat_15796,
        feat_11978`); the within-cluster pair `feat_7836 ↔
        feat_11978` (Polygram-predicted 0.987, real-decoder 0.992)
        is the natural choice. Optionally include one cross-cluster
        pair for contrast if the harness supports it with no extra
        effort.
      - **No φ optimization, no Dictionary baking, no Cancellation
        runs.** Polygram φ doesn't map to `W_dec`; this probe
        deliberately doesn't try to invert that. The probe is
        purely observational: does Polygram's predicted overlap
        line up with real-model co-firing and substitutability?
      Concrete plan:
      (a) Helper that loads GPT-2-small (via `transformers`) plus
          the SAE encoder/decoder weights (via the existing
          `load_sae_safetensors` path; the loader already reads
          `W_dec` and can be extended to also surface `W_enc` /
          `b_enc`).
      (b) Forward a held-out text batch (~1000 tokens from a
          fixed prompt set committed alongside the script) and
          collect SAE feature activations at layer 0 resid_pre
          for the chosen pair.
      (c) Compute three real-model statistics for the pair:
          - **Co-occurrence rate**: `P(B fires | A fires)` over
            the token set, with "fires" defined as activation
            above a fixed threshold (median or a fixed percentile
            of the per-feature activation distribution).
          - **Activation correlation**: Pearson correlation of
            the two features' raw activation values across all
            tokens.
          - **Substitutability**: per-token KL divergence
            between the model's next-token distribution under
            (1) baseline forward pass, (2) ablate-A (zero out
            feature A's activation post-encoder, before decoder),
            and (3) ablate-B. The pair substitutes if
            KL(ablate-A) ≈ KL(ablate-B) on the same tokens.
      (d) Compare against Polygram's predicted Gram entry for the
          pair (and against the decoder-cosine Gram already
          computed in PR #18); report whether high Polygram
          overlap coincides with high real co-occurrence and high
          substitutability, vs whether either real signal can
          differ sharply from Polygram's number.
      Three outcomes shape the next move:
      - All three real signals high (co-occurrence > 0.5,
        Pearson > 0.5, KL substitutability ratio in [0.5, 2.0]):
        Polygram's high-overlap classification predicts real
        feature redundancy. The remaining blockers in
        `spec-disentanglement-loop.md` (capability-preservation
        metric, differentiable `from_sae_lens`) become the next
        de-risking targets, not the basic premise.
      - At least one real signal contradicts Polygram (e.g., high
        Polygram overlap but low co-occurrence): decoder geometry
        and behavioural geometry are genuinely different — the
        peer-agent steering loop is investigating a misnamed
        signal. Polygram triage on `from_sae_lens` outputs needs a
        behavioural-data calibration step before any compression
        loop can ride on it.
      - Mixed (one of three signals low): write up which carries
        and which doesn't; update the practical-implications
        section of `decoder-gram-validity.md` accordingly.
      Ship as `docs/research/behavioural-gram-probe.md` plus the
      reproducible `examples/behavioural_gram_probe.py` script,
      same shape as the cross-encoding (PR #16) and decoder-gram
      (PR #18) spikes. Adds an optional `transformers` import path
      (best-effort import + skip if absent, matching the
      `safetensors` / `huggingface_hub` pattern). No new polygram
      surface required — the script reads the SAE checkpoint,
      loads GPT-2, runs forward passes, computes statistics.
      Blocks: any compression-pipeline or full disentanglement-loop
      work that wants to claim real-model grounding. The
      Gemma-Scope steering loop a peer agent drafted assumes high
      Polygram overlap → high real-model effect; this probe is the
      smallest test of that assumption.
      (Source: 2026-05-04 follow-up after PR #18 landed; the
      peer-agent "Spec-DisEntanglement Loop v0.1" draft
      pre-committed to numerical thresholds and assumed Gemma-Scope
      steering infrastructure that doesn't exist in this repo.
      Pushed back with a smaller scoped probe; user accepted.)

      *Closed 2026-05-04*. Empirical findings landed in
      `docs/research/behavioural-gram-probe.md` and the reproducible
      script `examples/behavioural_gram_probe.py`. TL;DR across the
      within-cluster pair `feat_7836 ↔ feat_11978`
      (Polygram-predicted 0.987, decoder 0.992) and the
      cross-cluster contrast `feat_7836 ↔ feat_15796`
      (Polygram-predicted 0.464, decoder 0.904), 654 held-out
      tokens: Polygram's *ordering* of pairs is preserved in real
      behaviour (within > cross on Jaccard 0.30 vs 0.19, Pearson
      +0.25 vs +0.19) but its *magnitudes* are heavily compressed
      (Polygram's 2.1× gap collapses to ~1.6× on Jaccard). Even the
      0.987-overlap pair fires together on only ~30% of token
      positions where either fires — high decoder overlap does not
      mean "always co-fires." Layer-0 ablation-KL is ~5e-5 for
      every feature: too small to discriminate, so the
      substitutability metric trivially passes. Outcome bucket:
      *mixed* — co-occurrence and Pearson carry a signed signal
      aligned with Polygram's prediction; ablation-KL is undefined
      at this layer. Practical implications: (a) Polygram is a
      ranker at the *behavioural* level too, not just for decoder
      geometry (corroborates §4.1); (b) "high decoder overlap" is
      not "redundant feature" — co-firing patterns matter; (c)
      layer-0 ablation-impact metrics measure noise, the
      peer-agent steering loop should target deeper SAE layers;
      (d) the §4.1 closure stands and is now confirmed at the
      behavioural level. Follow-ups: repeat at deeper layers
      (`blocks.5` / `blocks.10` resid_pre SAEs from the same
      `jbloom/...` repo); scale to 30+ pairs to measure the
      Polygram → behavioural-Jaccard correlation directly; replace
      next-token-KL with a logit-lens or attention-shift metric.

- [x] 4.3 Deeper-layer ablation-KL probe (research-track). PR #20
      settled the §4.2 question for one within-cluster pair plus a
      cross-cluster contrast, and surfaced an unexpected blocker for
      any future loop that wants to use ablation-KL as a behavioural
      impact signal: at `blocks.0.hook_resid_pre` on GPT-2 small,
      individual SAE feature ablations produce ~5e-5 nats of KL on
      the next-token distribution — too small to discriminate one
      feature from another, because eleven transformer blocks
      downstream compensate for any single feature's removal. The
      §4.2 substitutability metric trivially "passed" for both pairs
      because the underlying signal was indistinguishable from float32
      noise. This task is the smallest probe that tells us whether
      ablation-KL becomes informative deeper in the residual stack,
      or whether it is fundamentally weak for SAE features regardless
      of layer.
      Scope is deliberately narrow:
      - **Same model.** GPT-2 small, same prompt set as §4.2 (12
        multi-sentence paragraphs ≈ 654 tokens), same pair set
        (`feat_7836 ↔ feat_11978` within-cluster + `feat_7836 ↔
        feat_15796` cross-cluster contrast).
      - **Two new SAE checkpoints.** Same `jbloom/...` Hugging Face
        repo `jbloom/GPT2-Small-SAEs-Reformatted`, the
        `blocks.5.hook_resid_pre` and `blocks.10.hook_resid_pre`
        checkpoints. Same loader path that already worked for
        layer 0; lazy-import + skip-if-missing pattern unchanged.
      - **Same script, parameterized hook layer.** Add a
        `--layer {0,5,10}` flag to `examples/behavioural_gram_probe.py`
        (or equivalent) so the same harness moves the
        `forward_pre_hook` from `model.transformer.h[0]` to
        `model.transformer.h[5]` / `model.transformer.h[10]` and
        reads the corresponding SAE checkpoint. No new statistics —
        the existing co-occurrence / activation-Pearson / paired
        ablation-KL battery is the right shape, this probe just
        re-runs it at depth.
      - **No φ optimization, no new feature subsets, no Gemma.**
        Co-firing and Pearson are layer-specific properties (a
        feature's encoder reads from that layer's residual stream),
        so they will move; ablation-KL is the load-bearing
        measurement here. Report all three for completeness.
      Concrete plan:
      (a) Extend the §4.2 script with a `--layer` flag and matching
          checkpoint resolver.
      (b) Run at `blocks.5` and `blocks.10`; record per-pair
          co-occurrence Jaccard, activation Pearson, and the
          paired-on-both-fire ablation-KL ratio plus absolute KL
          magnitudes for both ablations.
      (c) Plot/table KL magnitude vs layer (0, 5, 10) for each
          feature in the pair, and the substitutability ratio at
          each depth.
      Three outcomes shape the next move:
      - **KL grows monotonically with depth** (e.g., 5e-5 → 1e-3 →
        1e-2 nats): ablation-KL becomes informative deeper in the
        stack. Future loops that need a behavioural-impact signal
        should target the deepest layer where the SAE is still
        well-conditioned (likely `blocks.10`). The peer-agent
        compression-loop sketch can use ablation-KL as drafted, but
        only if the layer is moved.
      - **KL stays flat across layers** (still ~10⁻⁵ nats at
        `blocks.10`): ablation-KL is fundamentally weak for
        single-SAE-feature interventions on GPT-2 small. Loops need
        a different impact signal — logit-lens shift, attention
        pattern divergence, or activation-norm change at a
        downstream layer — before any compression objective can use
        "behavioural impact" as a term.
      - **Non-monotonic** (e.g., `blocks.5` larger than `blocks.10`,
        or signal concentrated at one layer): write up which layer
        carries signal and why; the answer constrains layer choice
        for any future Polygram-driven steering work and rules out
        naive "deeper is always better."
      Ship as `docs/research/deeper-layer-ablation-probe.md` plus
      the extended `examples/behavioural_gram_probe.py` (same
      reproducible-script shape as §4.1 / §4.2). Smoke test in
      `tests/test_examples.py` parallels the §4.2 smoke test —
      `--layer 0 --n-prompts 1 --quiet` keeps the test cheap; full
      runs are optional / opt-in via the existing skip path when the
      checkpoint isn't downloaded.
      Blocks: any compression-loop or disentanglement-loop spec that
      wants to use ablation-KL (or any next-token-impact metric) as
      part of its objective. Together with §4.4 (scale-up to 30+
      pairs) this closes the cheapest open questions before a full
      loop spec can be justified.
      (Source: 2026-05-04 follow-up after PR #20 landed §4.2;
      ablation-KL came in three orders of magnitude smaller than
      anything that could discriminate between features, making the
      §4.2 substitutability metric trivially pass and forcing this
      explicit follow-up before that metric — or any descendant of
      it — is wired into a real loop.)

      *Closed 2026-05-04*. Empirical findings landed in
      `docs/research/deeper-layer-ablation-probe.md` and the extended
      `examples/behavioural_gram_probe.py --layer {0,5,10}`. TL;DR
      across the same 654-token batch and the same arbitrary anchor
      features at three layers: per-feature ablation-KL on the
      next-token distribution jumps from ~5e-5 nats at `blocks.0` to
      ~1.04–1.93 nats at `blocks.5`, then plateaus at ~0.56–2.04 nats
      at `blocks.10`. Outcome bucket: *monotonic with plateau* — the
      0 → 5 transition spans roughly four orders of magnitude; the
      5 → 10 transition is essentially flat. Layer 0 is the structural
      dead zone (eleven downstream blocks compensate for any single
      input-layer feature's removal); layers 5 and 10 are roughly
      equivalent for ablation-impact purposes. Practical implications:
      (a) any future compression / disentanglement loop that wants to
      use ablation-KL (or any next-token-impact metric) should hook at
      `blocks.5` or deeper on GPT-2 small; (b) the peer-agent
      `compression_score` sketch is unblocked at depth, blocked at
      layer 0; (c) `blocks.5` ≈ `blocks.10` for ablation magnitude,
      so layer choice between the two is driven by other concerns
      (compute cost vs proximity to unembedding). Caveats: feature
      ids `[7836, 11978, 15796]` were chosen on the layer-0 SAE; at
      layers 5 / 10 they index *different* SAE features. The
      per-feature KL magnitudes (the load-bearing finding) survive
      this caveat — the 4-OOM gap is the signal — but pair-level
      co-occurrence / Pearson / substitutability metrics at depth
      are not comparing the same semantic pair as layer 0 and should
      not be read as cross-layer behavioural claims about specific
      Polygram pairs. Pearson +1.000 at layers 5 and 10 with
      near-total co-fire signals near-degenerate firing patterns for
      these arbitrary indices, not a property of deep-layer SAE
      features in general. Follow-ups: layer-local feature selection
      (apply projection-similarity within each layer's own SAE
      separately) plus the §4.4 scale-up (30+ pairs at varied
      Polygram overlaps, picked per-layer) close the remaining
      cheap probes before a full loop spec.

- [x] 4.4 Polygram → behavioural-Jaccard correlation at scale
      (research-track). PRs #20 (§4.2) and #23 (§4.3) settled two
      load-bearing questions on a single within-cluster pair plus a
      cross-cluster contrast: Polygram's *ordering* (within > cross)
      survives at the behavioural level, and ablation-KL becomes
      informative at `blocks.5+` on GPT-2 small. What's still open
      is the *shape* of the Polygram → behavioural correspondence
      across many pairs. PR #20 had N = 2 pairs — too few to fit a
      correlation slope. The §4.1 closure measured Spearman 0.94
      between Polygram's predicted Gram and the *decoder* squared-
      cosine Gram on the Real SAE; this task asks the same question
      at the *behavioural* level. The natural metric is
      `Spearman(Polygram_overlap, Jaccard_co_fire)` across 30+ pairs
      drawn from a layer-local feature selection at a depth where
      ablation-KL is also usable.
      Scope:
      - **Same model.** GPT-2 small + the same `jbloom/GPT2-Small-
        SAEs-Reformatted` SAE family.
      - **One layer.** `blocks.10.hook_resid_pre` (closer to the
        unembedding than `blocks.5`, less downstream compensation;
        §4.3 found the two layers' ablation magnitudes are
        comparable so this picks the simpler-to-interpret one).
        Single-layer scope keeps the cost bounded; cross-layer
        slopes are a §4.5+ question if needed.
      - **Layer-local feature selection.** Apply the §4.1
        projection-similarity selection within `blocks.10`'s own
        SAE (not the layer-0 selection that PRs #16/#18/#20/#23
        reused). Pick a feature subset of size ~20–30 stratified
        across the Polygram-predicted overlap distribution: roughly
        equal counts in the low (≤ 0.4), medium (0.4 – 0.7), and
        high (≥ 0.7) MPS-overlap buckets. The pair count from such
        a subset is `n*(n-1)/2`, so 25 features → 300 pairs (well
        above the 30+ floor).
      - **Same prompt set, same statistics.** 12 paragraphs,
        ~654 tokens (the §4.2 / §4.3 batch). Per pair: Polygram
        predicted overlap, decoder squared cosine, Jaccard
        co-fire, activation Pearson, paired ablation-KL ratio on
        both-fire tokens (only computed for pairs with ≥ 5
        both-fire tokens — pairs that never co-fire don't have
        substitutability defined).
      - **No new statistics.** The §4.2 battery is the right
        shape. This task is purely about scale and per-layer
        feature selection — building a scatter, not building a
        new metric.
      Concrete plan:
      (a) New script `examples/behavioural_gram_scaleup.py` (kept
          separate from the §4.2 / §4.3 single-pair probe to avoid
          cluttering its arg surface). It loads the layer-10 SAE,
          builds the layer-local feature subset via projection
          similarity (reusing whatever path PR #16 / PR #18 used,
          parameterized to a chosen layer), constructs the Polygram
          dictionary via `from_sae_lens`, and computes pairwise
          predicted overlaps. Stratified sampling picks ~25
          features.
      (b) Forward the prompt set once, capture the residual stream
          at `blocks.10`, encode through the SAE → per-feature
          activations across 654 tokens.
      (c) For each of the ~25 selected features, run one ablation
          forward pass (subtracting that feature's decoder
          contribution at every token where it fires) → per-token
          KL on next-token distribution. Total ~25 ablation passes,
          ~10–15 minutes on CPU at 12 prompts.
      (d) Build the per-pair scatter: for each of the ~300 pairs,
          collect (Polygram_overlap, decoder_overlap, Jaccard,
          Pearson, KL_ratio_distance_from_1, n_both_fire). Report:
          - Spearman + Pearson correlations between Polygram and
            each behavioural metric across the pair set.
          - Same correlations between *decoder cosine* and each
            behavioural metric — gives us the "ceiling" (how much
            Polygram-vs-behaviour correlation is bounded by
            decoder-vs-behaviour correlation, since Polygram only
            sees decoder geometry).
          - Per-bucket means: Jaccard mean for low / medium / high
            Polygram-overlap pairs, with the 95% bootstrap CI on
            the gap.
      (e) Save the pair-level CSV alongside the research note for
          inspection / re-analysis.
      Three outcomes shape the next move:
      - **High Spearman (≥ 0.6) Polygram ↔ Jaccard.** Polygram's
        ranking transfers cleanly to behavioural co-firing at
        scale. The loop spec proceeds with Polygram as the
        primary candidate filter, `Jaccard ≥ τ` as a secondary
        gate (τ chosen from the per-bucket means), ablation-KL at
        `blocks.10` as the impact metric (from §4.3). Three
        constraints are set; spec is ready to write.
      - **Medium Spearman (0.3 – 0.6) Polygram ↔ Jaccard.**
        Polygram still ranks pairs above-chance but the slope is
        shallow enough that loops need an explicit calibration
        step: the loop has to measure per-workload Jaccard before
        committing to which pairs to compress. Loop spec proceeds,
        with a calibration phase added before any candidate is
        actioned.
      - **Low Spearman (< 0.3) Polygram ↔ Jaccard.** Polygram's
        decoder-geometry ranking does not transfer to behavioural
        co-firing at this layer. Either the layer-10 SAE encodes
        in a way decoupled from layer-0-style decoder cosines
        (testable by also reporting decoder ↔ Jaccard correlation
        — if that's also low, the gap is fundamental), or
        Polygram's `from_sae_lens` projection compresses too
        aggressively at this scale. Loop spec is blocked until
        either a per-layer-aware Polygram parameterization or a
        non-Polygram candidate-selection signal is built.
      Ship as `docs/research/behavioural-scaleup-probe.md` plus
      `examples/behavioural_gram_scaleup.py` plus a pair-level CSV
      `docs/research/data/scaleup_pairs.csv`. Smoke test in
      `tests/test_examples.py` parallels §4.2 / §4.3: invoke with
      a tiny `--n-features 4 --n-prompts 1 --quiet` configuration
      so the test stays cheap and exercises both the
      checkpoint-present path and the checkpoint-missing skip.
      Blocks: the full compression / disentanglement loop spec.
      §4.3 settled the layer choice; §4.4 settles whether
      Polygram's selection signal is usable at scale. After both,
      the loop spec writes itself: known layer (`blocks.10`),
      known calibration slope (from §4.4), Polygram-as-ranker
      with the §4.4 Jaccard threshold as a co-firing gate,
      ablation-KL at `blocks.10` as the impact term.
      (Source: 2026-05-04 follow-up after §4.3 closed the
      layer-choice question. The §4.2 / §4.3 single-pair contrast
      established the directional signal but is statistically
      anecdotal at N = 2; §4.4 is the smallest probe that turns
      the directional claim into a calibrated slope.)
      Done 2026-05-05 in `add-behavioural-scaleup-impl`. Scope
      ran at the cap-imposed sizing (8 features → 28 pairs)
      rather than the spec's "~25 features → ~300 pairs" — the
      rung-1 MPS encoding caps a Dictionary at 8 features
      (`MAX_FEATURES_PER_DICTIONARY` in `polygram/sae_import.py:23`),
      a constraint the spec missed at draft time. Headline
      result clears the 0.6 threshold:
      `Spearman(Polygram, Jaccard) = +0.637`. Per-bucket Jaccard
      means separate cleanly: mid-overlap (0.4–0.7, n=16)
      Jaccard 0.145 [0.10, 0.19] vs high-overlap (≥0.7, n=12)
      Jaccard 0.621 [0.43, 0.82] — non-overlapping 95% CIs.
      Decoder-cosine alone gives Spearman −0.054 against
      Jaccard at this layer (selection-conditional finding —
      seed-stratified bimodal cosine distribution kills decoder-
      Jaccard variation, but Polygram's γ-spread still ranks).
      Three caveats survive: low-overlap bucket is empty inside
      the cap (β-spread between 2 KMeans clusters floors
      cross-cluster squared overlap at ~0.44, above the 0.4
      bucket boundary); 28 pairs share a single seed feature so
      effective N for bootstrap CIs is overstated; ablation-KL
      ratio gives a weaker (Spearman −0.33) confirmatory signal
      not a primary one. Outcome bucket: high Spearman, loop
      spec unblocked. Loop constraints settled: hook at
      `blocks.10`, Polygram-as-primary-ranker, `Jaccard ≥ 0.30`
      as required co-firing gate (chosen between the bucket
      means' CIs), ablation-KL at `blocks.10` as
      per-feature impact metric. Shipped:
      `examples/behavioural_gram_scaleup.py`,
      `docs/research/behavioural-scaleup-probe.md`,
      `docs/research/data/scaleup_pairs.csv`,
      `docs/research/data/scaleup_probe_full.log`,
      smoke test in `tests/test_examples.py`.

## 5. Compression / disentanglement loop — first half: validator

- [ ] 5.1 The "loop spec writes itself" sentence in §4.4's closure
      block (above) names four settled constraints — `blocks.10` hook,
      Polygram-as-primary-ranker, `Jaccard ≥ 0.30` co-firing gate,
      ablation-KL at `blocks.10` as the per-feature impact term. The
      first half of the loop is the **read-only validator** that runs
      the four-constraint pipeline against any user-supplied
      Dictionary and emits a structured `ValidationReport`. The second
      half (the weight-modifying compression action) consumes the
      validator's confirmed-candidate report.
      Shipped as `add-behavioural-validator-loop` (PR #26 spec,
      implementation follow-up): new `polygram.behavioural`
      subpackage hosting `BehaviouralValidator`, `ValidationReport`
      JSON + CSV round-trip, `polygram validate` CLI subcommand,
      optional `[behavioural]` extra. The compression action is the
      next change after this one.
- [ ] 5.2 Compression-action default encoding is **gated on the
      Rung3 §4.5 viability spike** (`add-rung3-encoding-mvp`,
      PR #29 → impl PR pending). Rung3 ships `Cancellation(
      encoding="rung3")` and the joint
      (φ, θ_amp, ψ_aux) optimizer that, in principle, breaks below
      MPSRung1's phase-only floor. The follow-up findings PR
      (`docs/research/rung3-viability-spike.md`) decides:
      - **strong-pass** → open `make-rung3-default` change flipping
        the production encoding for `Dictionary` / `Cancellation` /
        `BehaviouralValidator` consumers, then implement compression
        on Rung3.
      - **partial / fail** → compression-impl proceeds with
        MPSRung1 as default; Rung3 stays opt-in.
      The compression spec itself (`add-compression-action`, PR #28)
      is encoding-agnostic and lands independently — only the
      compression-impl PR needs to wait for the verdict.

      *Resolved 2026-05-05*. The §4.5 spike returned `partial_pass`
      under the non-degenerate-amp constraint
      (`docs/research/rung3-viability-spike.md` — the unconstrained
      `strong_pass` was a trivial amp-zeroing degeneracy; the
      constrained re-run binds at `floor × ε` on every pair).
      `make-rung3-default` is dead. Compression-impl proceeds with
      **MPSRung1 as the production default**; Rung3 ships as
      opt-in via `Cancellation(encoding="rung3")`. The compression
      action itself does not select an encoding — it operates on
      raw decoder/encoder weights and is encoding-agnostic by
      construction.

- [x] 5.3 Compression-action implementation. Spec landed in
      `add-compression-action` (PR #28); implementation ships the
      `polygram.compression` subpackage with `Compressor`,
      `CompressionPlan`, `CompressionReport`, `CompressionResult`,
      and `ClusterPlan`; the `zero` strategy (encoder column +
      bias + decoder row, leaving `b_dec` untouched); the
      `polygram compress` CLI subcommand; round-trippable JSON
      report carrying source/output sha256s + upstream
      `ValidationReport` provenance; worked example
      `examples/compress_validated.py`; full test ladder under
      `tests/compression/` plus CLI tests under
      `tests/cli/test_compress_cli.py` and a smoke test in
      `tests/test_examples.py`. README "Compression action"
      section + `docs/research/compression-action-design.md`
      pointer note added. Component-first via union-find on
      `validation_report.confirmed` makes the operation
      order-independent; the spec's far-cluster clique observation
      from the §5.1 live run forced this shape.
