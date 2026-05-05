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
