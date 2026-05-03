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
