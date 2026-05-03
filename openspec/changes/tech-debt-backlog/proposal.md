## Why

We keep turning up small items during code review and QA triage that
aren't worth their own OpenSpec change — README polish, naming nits,
docstring gaps, error messages that could be clearer, leftover dead
code — but they also shouldn't disappear into commit-message limbo.

Mirrors the rolling-backlog pattern already in use upstream in
`q-orca-lang`.

## What Changes

- Create a long-lived "backlog" change that collects these small
  items in one `tasks.md`, grouped by area.
- Items are added as they come up (from QA triage, self-review, PR
  review). Once an item is fixed, its task is marked complete but
  **not** removed — the eventual archive preserves what was done and
  why.
- When an item grows enough to warrant its own proposal/design
  (behavior change, new requirement, cross-module impact), it is
  pulled out into a dedicated OpenSpec change and the backlog entry
  is marked with a pointer to that change.
- This change is intentionally long-lived: it stays open until it
  gets too big to manage, at which point we archive it as a snapshot
  and start a fresh one.

## Impact

- Affected specs: **none** — this is a process/backlog change, not a
  requirements change. No `specs/` deltas (intentionally fails
  `openspec validate --strict`; track strict only on real spec
  changes).
- Affected code: items land as small PRs referencing their task
  number in `tech-debt-backlog/tasks.md`.
