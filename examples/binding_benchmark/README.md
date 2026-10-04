# Binding-preservation study

Research protocol: the archived `binding-preservation-benchmark` OpenSpec change.
Capability requirements: `openspec/specs/binding-preservation/spec.md`.
Frozen run configuration: `runs/binding/protocol.json`.

This benchmark measures subject/object, variable/value, and explicit modifier
bindings. Every minimal pair has identical lexical content, opposite target
answers, and an unchanged distractor answer. Combination holdouts reserve known
fillers in new roles; separate tests introduce unseen templates. A development
partition is used only for host selection.
These combinations are withheld from benchmark fitting, not from the pretrained
host's training history.

The initial study replaces the final pre-RMSNorm residual at the last prompt
position. Only the host's RMSNorm and unembedding follow it, so untouched earlier
positions cannot bypass the replacement via attention. This is a task-conditioned
readout experiment, not evidence about how earlier layers compute the bindings.

## Run sequence

From the Polygram repository root, with the behavioural extras installed:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m examples.binding_benchmark.host \
  --model Qwen/Qwen2.5-1.5B-Instruct --dtype bfloat16 \
  --output runs/binding/dev-v2-qwen15

HF_HUB_OFFLINE=1 .venv/bin/python -m examples.binding_benchmark.capture \
  --protocol runs/binding/protocol.json --output runs/binding/capture-fit

.venv/bin/python -m examples.binding_benchmark.representations \
  --capture runs/binding/capture-fit --output runs/binding/fits --seed 0
```

Repeat fitting with seeds 1 and 2. All method selection uses training and
validation only. For a fresh run, freeze fit hashes before starting final capture
(the recorded run already has this manifest; do not overwrite it):

```bash
.venv/bin/python - <<'PY'
import datetime
import hashlib
import json
from pathlib import Path

root = Path("runs/binding/fits")
manifest = root / "frozen.json"
assert not manifest.exists(), "Do not replace an existing freeze"
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
payload = {
    "frozen_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "files": {str(p): digest(p) for p in sorted(root.iterdir()) if p.is_file()},
    "source_sha256": {
        str(p): digest(p) for p in sorted(Path("examples/binding_benchmark").glob("*.py"))
    },
}
manifest.write_text(json.dumps(payload, indent=2) + "\n")
PY
```

After every fit is frozen:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python -m examples.binding_benchmark.capture \
  --protocol runs/binding/protocol.json --output runs/binding/capture-test \
  --splits test_combinations,test_templates

.venv/bin/python -m examples.binding_benchmark.evaluate \
  --capture runs/binding/capture-test --fits runs/binding/fits/fits-seed0.json \
  --output runs/binding/eval-seed0
```

Repeat evaluation for every fitted seed. Run `examples.binding_benchmark.forge`
in an environment with the sibling sae-forge package installed, using the saved
compressed checkpoint and the same final capture:

```bash
HF_HUB_OFFLINE=1 python -m examples.binding_benchmark.forge \
  --capture runs/binding/capture-test \
  --checkpoint runs/binding/fits/compressed-target32-seed0.safetensors \
  --output runs/binding/forge-target32-seed0
```

The forge is explicitly native-in-basis, saved and reloaded before evaluation.
It takes token IDs without teacher hooks. Captured host residuals are used only
to calculate reference metrics. The same basis is also tested as a linear
projection at the final residual; no host-wrapped fallback is allowed.

## Interpretation

- Candidate-answer accuracy is discrimination among three explicit alternatives.
  It is not unrestricted generation accuracy. Development reports also include
  unrestricted first-token accuracy.
- The TPR is supervised by role annotations and its edits need no activation donor.
  SAE and subspace edits use counterfactual activation donors; these are separate
  intervention information budgets.
- Unrelated-query edits are compared against the original host; target-query edits
  against the actual counterfactual host. No-op and identity controls are included.
- Full-vocabulary KL is in nats. Pair accuracy requires both members to be correct.
  Intervals resample pairs, not individual members. Host failures remain in all
  reports, with reference-correct conditional accuracy alongside unconditional scores.
- Polygram's actual target-K merge ranks training co-firing Jaccard. It has no
  causal redundancy confirmation here. Actual kept width/rank is reported even
  if the requested target is not reached, with matching PCA/learned controls.
- Representation checkpoints, parameter counts, split hashes, per-example metrics,
  and negative outcomes are retained. Large binary arrays/checkpoints are locally
  saved and gitignored; JSON manifests and reports are reviewable research artifacts.

## Current evidence

The first 0.5B host screen failed two task gates. Dataset v2 uses single-token
answer symbols; Qwen2.5-1.5B-Instruct passed all three development gates at 100%
candidate and both-members accuracy (12 pairs per task). These small development
screens are host-selection evidence only. The completed three-seed findings are in
`docs/research/binding-preservation-results.md`, with the five-requirement completion
audit in `docs/research/binding-preservation-status.md`.

After all three evaluation and forge reports exist, regenerate the tables with:

```bash
.venv/bin/python -m examples.binding_benchmark.report \
  --output docs/research/binding-preservation-results.md
```

The recorded run includes `sae-training-diagnostics.json`, `evaluation-provenance.json`,
`checks.json`, and `completion-audit.json` under `runs/binding/`.
