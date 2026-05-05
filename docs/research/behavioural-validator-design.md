# Behavioural validator — design note

> One-page pointer for the `polygram.behavioural.BehaviouralValidator`
> change. The full spec lives in
> [`openspec/changes/add-behavioural-validator-loop/`](../../openspec/changes/add-behavioural-validator-loop/).

## What the validator is

`BehaviouralValidator` runs the four-constraint compression-loop
pipeline against a `Dictionary` of SAE features and emits a
structured `ValidationReport`. The four constraints are the four
shipped probes:

- **§4.1 / PR #18** — Polygram is a ranker, not a magnitude predictor
  ([`decoder-gram-validity.md`](decoder-gram-validity.md)).
- **§4.2 / PR #20** — high decoder overlap ≠ behavioural redundancy
  ([`behavioural-gram-probe.md`](behavioural-gram-probe.md)).
- **§4.3 / PR #23** — ablation-KL is a structural dead zone at
  `blocks.0.hook_resid_pre` on GPT-2 small; hook at `blocks.5+`
  ([`deeper-layer-ablation-probe.md`](deeper-layer-ablation-probe.md)).
- **§4.4 / PR #25** — `Spearman(Polygram_overlap, Jaccard_co_fire)`
  reaches +0.637 at `blocks.10`, above the 0.6 "loop unblocked"
  threshold ([`behavioural-scaleup-probe.md`](behavioural-scaleup-probe.md)).

The validator's API is two-stage:

- `predict()` — Polygram-only, no torch needed. Returns one
  `CandidatePair` per `(i, j)` with `polygram_overlap` and
  `decoder_overlap` populated; behavioural fields are NaN.
- `validate(candidates=None)` — lazy-imports torch + transformers,
  runs `len(feature_ids)` ablation forward-pass-batches, and
  populates the per-pair Jaccard / Pearson / KL fields. The cost cap
  (≤ N ablation batches; never N²/2) is encoded as a spec contract.
- `run()` — `validate(predict())`.

## Why this scope was the right next thing

After §4.4 settled the four constraints empirically, the natural
question was "what ships next?" Two candidates:

1. **The validator** (this change). A read-only library surface that
   runs the four-constraint pipeline against any user-supplied
   Dictionary and emits a structured report.
2. **The compression action**. A weight-modifying merge / zero-out
   pass that consumes Polygram-flagged candidates that survive the
   behavioural gate.

Compression's blast radius is much larger: it writes back to `W_dec`,
which means every downstream evaluation gets re-derived against the
modified weights. The validator's blast radius is zero — it only
*reports*. Shipping the validator first means:

- Anyone running compression downstream can independently audit the
  upstream candidate report by replaying the validator on the same
  inputs. Without the validator surface, the upstream signal lives in
  a single-purpose `examples/behavioural_gram_scaleup.py` script and
  can't be replayed against arbitrary Dictionaries.
- The validator's tight contract (§4.4 thresholds as defaults,
  per-feature ablation cap, layer-0 refusal) means future researchers
  can't accidentally drift away from the constraint chain.
- Compression depends on the validator's report shape; defining the
  shape first forces compression's interface to be straightforward.

## What the validator deliberately does not do

The validator does not modify SAE weights. It does not train,
fine-tune, or backprop through anything. It does not auto-tune
thresholds. It does not stitch multiple Dictionaries for >8 features.
Each of those is a separate change, sized appropriately.

## See also

- `examples/behavioural_validate.py` — worked example: build a
  Dictionary from the §4.4 selection, run the validator, dump
  JSON + CSV.
- `polygram validate` — CLI subcommand wrapping `run()` with file-
  based inputs.
