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

- [ ] 3.1 Encoding-invariance spike (MPS vs HEA classification
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
