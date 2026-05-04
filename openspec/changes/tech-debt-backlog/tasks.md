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

- [ ] 2.1 Per-knob HEA floor diagnostics (research-track). Add
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
