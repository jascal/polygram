# HEA_Rung2 structural floor — what exists, what doesn't, and why

> Research-track note addressing the deferral in
> [`Cancellation.structural_floor()`](../../polygram/cancellation.py)'s
> docstring and raised `NotImplementedError`: for any HEA-encoded
> dictionary, "a defensible HEA bound... is deferred to a follow-up
> research-track proposal." (The README's
> ["Choosing an encoding"](../../README.md#choosing-an-encoding) section
> states the same conclusion more briefly, without that phrase.) This is
> that investigation. It used
> an external tool, [`lagh`](https://github.com/jascal/lagh) — a certified
> symbolic-law-discovery engine (certificate-or-reasoned-abstention, never
> a confident guess) — cross-checked against the actual gate sequence in
> `q_orca.compiler.concept_gram_hea`. Every numeric result below is
> reproducible from the snippets in
> [Reproducing the result](#reproducing-the-result).

## TL;DR

**No general closed-form floor exists for HEA's full multi-knob case —
and this is now proven, not assumed.** But several *narrower* cases
provably do have closed forms, some strong enough for a real `lagh`
certificate:

| Case | Knobs varied | Closed form? | `lagh` certificate |
|---|---|---|---|
| Single knob, `φ` (layer 0) | 1 | yes — `M − V·cos(φ)` | **pinned**, α ≤ 1e-64.56 |
| Single knob, `theta[Ry,0,0]` (layer 0) | 1 | yes — `M(1+cos θ)` | **pinned**, α ≤ 1e-21.87 |
| Single knob, `theta[Ry,1,0]` (layer 1, post-entangler) | 1 | yes — needs a `sin` term too | **pinned**, α ≤ 1e-125.87 |
| Same slot, shared across the target pair (layer 0) | 2 | yes — collapses to `M(1+cos Δθ)` | **pinned**, α ≤ 1e-508 (via `verify`) |
| Two *different* slots, same feature (layer 0 + layer 1) | 2 | yes — full 9-term basis, no collapse | none — `recover` abstained through all 7 tiers; form proven by circuit theory, confirmed to 5.6e-16 residual, but `verify` won't pin non-rational coefficients |
| Three independent slots (2 features) | 3 | yes — 8/27-term basis | not attempted — pure circuit-theory + curve-fit |
| All 24 knobs (12/feature × 2 features) | 24 | **provably exists, in a `3²⁴`-term space; not usefully computable** | not attempted — `fit` timed out past 300s even as an unbounded scout |

The general theorem (below) explains every row at once: each knob is a single-qubit rotation parameter, applied exactly once, so the squared overlap lives in a `3^N`-dimensional trig-polynomial space. Whether that space collapses to something small is a case-by-case question about circuit symmetry, not something `lagh`'s registered curriculum can answer unaided past `N=1` — its trig-term library has no cross term for two independent angular columns.

## Method

Two complementary tools, used in the order `lagh`'s own docs recommend when its bounded tools (`recover`/`verify`) abstain:

1. **`lagh.recover(X, y, sigma=0)`** on real, deterministic `dictionary.gram()` output (`sigma=0` because these are exact analytic Gram computations, no measurement noise) — either a certificate (`pinned`/`consistent`, with a significance bound `α ≤ |H|·q^h`) or a structured abstention.
2. When `recover` abstains, its own diagnosis (`"a periodic component is likely, declare_and_verify"`) is the cue to derive the actual functional family from the circuit's gate structure, then hand the resulting hypothesis to **`lagh.verify(X, y, form, sigma)`** for an honest check.

`lagh`'s tier-1 curriculum (`c1_polynomial.py`) builds trig-monomial features for a *single* angular column at a time (`sin^a(x_k)cos^b(x_k)`, optionally times a linear — not trig — term in another column); no class in its curriculum builds `cos(x_i)·cos(x_j)` for two independent angular columns `i≠j`. That gap, not any property of HEA, is why `recover` cannot find the multi-knob results below on its own — confirmed empirically (see [Caveats](#caveats)).

## Single-knob floors — the entangler changes the family

Fixing the target pair's *other* feature at the slot's default/zero and sweeping one knob on the swept feature over its full range:

| Knob | Layer | Certified law | Notes |
|---|---|---|---|
| `φ` | 0 | `441642253943/500000000000 − (5734592931/50000000000)·cos(φ)` | Same family as `MPSRung1` (see [`cancellation-phase-floor.md`](cancellation-phase-floor.md)) — pure cosine, floor at `φ=0`. |
| `theta[Ry,0,0]` | 0 | `(385075576467/1000000000000)·(1 + cos θ)` | Pure cosine again; floor hits **exactly 0** at `θ=π`. |
| `theta[Ry,1,0]` | 1 (post-entangler) | `38429919657/100000000000 + (76858690539/200000000000)cos θ − (1050559399/500000000000)sin θ` | A genuine `sin` term appears. Floor is no longer at `δ=π` — it's phase-shifted, landing at `M − √(A²+B²) ≈ 2.9e-6`. |

The pattern: knobs at layer 0 have been through **two** rounds of the ring entangler by the time the circuit finishes (its own layer's entangler, then layer 1's); a layer-1 knob only goes through **one**. The extra round is what introduces the `sin` term — i.e. the closed form for a single HEA knob generalizes from `MPSRung1`'s `M ± |V|` to `M ± √(A²+B²)`, with the simpler case recovered when the entangler-round count happens to leave a residual symmetry.

## Two knobs, shared slot across the pair — still collapses

Sweeping the *same* slot (`theta[Ry,0,0]`) independently on both features of the target pair, as a raw 2-D grid (`θ_A`, `θ_B` as separate columns, no hint that only the difference matters):

- `recover` on the raw grid: **abstain**, `"non-algebraic... periodic component likely"`, `next_action: declare_and_verify`.
- Declaring `M(1+cos(θ_A−θ_B))` (the form already pinned from the single-knob case) and handing the **full, un-reduced** 225-point grid to `verify`: **certified, pinned, α ≤ 1e-508** — the tightest bound in this whole investigation, and confirmation that `recover`'s abstain was a curriculum gap, not a real absence of structure.

## Two knobs, different slots, one feature — genuinely doesn't collapse

Sweeping `theta[Ry,0,0]` and `theta[Ry,1,0]` — layer 0 and layer 1, different rounds of entangling — both on `bird_hawk` only (`dog_poodle` fixed):

- Neither `θ_1 − θ_2` nor `θ_1 + θ_2` reduces the 2-D grid to any 1-D function (spread ≈ full data range for both hypotheses).
- `recover` on the raw grid ran for ~10–15 minutes, exhausted **all seven** of `lagh`'s registered tiers, then abstained with the same "periodic, declare_and_verify" diagnosis — genuinely harder than the shared-slot case, not just slower.
- No obvious substitution to declare this time. Circuit theory instead: each knob parameterizes exactly one `Ry` gate, applied once, so by the argument in the next section the squared overlap must live in the 9-term basis `{1,cosθ_1,sinθ_1}⊗{1,cosθ_2,sinθ_2}`. Fitting that basis by ordinary least squares against the 225-point grid: **rank 9/9, residual 5.6e-16** — the theory is exactly right, and this time genuinely needs the full basis, no collapse.
- Handing the fitted form to `verify` did **not** produce a certificate: at `sigma=0` it's refuted by ~1e-13 (an artifact of my float64→decimal-string round-trip, not a real mismatch); at a realistic `sigma=1e-10` it abstains `"parametric"` — *"coefficient not pinned: a perturbed refit scale also certifies."* This is `verify` working as designed: its exact-coefficient gate is built for one hypothesis with at most one free scale, not an empirically-fit 8-coefficient regression. The coefficients here are genuine irrational numbers (functions of the other ~10 fixed circuit angles on both features), so there's no clean rational for it to pin.

## The general N-knob theorem

**Claim.** If `N` distinct parameters each control exactly one single-qubit rotation gate, applied exactly once, anywhere across both circuits, then

```
overlap(θ_1,...,θ_N) ∈ span{ ∏ᵢ fᵢ(θᵢ) : fᵢ ∈ {1, cos θᵢ, sin θᵢ} }
```

— a `3^N`-dimensional space, degree ≤1 in each variable separately.

**Proof sketch.** `Ry(θ)`/`Rz(θ)` are each linear in `{cos(θ/2), sin(θ/2)}`. Gate composition is matrix multiplication (linear), so it can't raise the degree in `θᵢ` beyond how many times that gate occurs — here, exactly once, confirmed directly from the gate-application loop in `q_orca.compiler.concept_gram_hea._build_hea_state` (`for layer: for rotation: for qubit: apply once`). So each of `|A⟩`, `|B⟩` is multilinear (degree ≤1 in each `θᵢ`'s half-angle pair, independently). `⟨A|B⟩` is a sum of per-basis-state products of one A-amplitude and one B-amplitude — summing doesn't raise degree, so it stays multilinear too. Squaring to `|⟨A|B⟩|²` doubles the half-angle degree to ≤2 per variable, and the double-angle identities (`cos²(θ/2)=(1+cosθ)/2` etc.) fold that back to full-angle degree ≤1. Do this independently per variable and the tensor product gives `3^N` terms, no more.

Confirmed at every `N` tested — always a valid upper bound, sometimes tight, sometimes not:

| N | bound | observed rank | tight? |
|---|---|---|---|
| 1 (layer 0 knobs) | 3 | 2 | no — extra symmetry kills the `sin` term |
| 1 (layer 1 knob) | 3 | 3 | **yes** |
| 2 (shared slot, cross-feature) | 9 | 2 | no — collapses to `M(1+cos Δθ)` |
| 2 (different slots, same feature) | 9 | 9 | **yes** |
| 3 (2 slots one feature + 1 on the other) | 27 | 8 | no — partial collapse |

Layer-1-only (post-entangler) knobs trend toward full rank; layer-0 knobs and cross-feature-shared knobs trend toward collapse. That's a real pattern, not yet a proven rule.

## Why the N=24 case resists a shortcut

`3^24 ≈ 2.8×10^11` terms — the theorem proves the true law lives *somewhere* in that space, but that's not a usable closed form. The natural shortcut is to ask whether the space is actually **sparse** — whether few-variable ("low interaction order") terms dominate, as they do for many bounded-depth circuits. Tested directly: 4000 random samples across all 24 knobs (12 per feature × 2 features, fully independent), fit against degree-truncated sub-bases:

| truncation | # terms | R² | "large" coefficients (>1e-3) |
|---|---|---|---|
| degree ≤0 (constant) | 1 | 0.0% | 1/1 |
| degree ≤1 (single-knob effects only) | 49 | 1.3% | 31/49 |
| degree ≤2 (pairwise cross terms) | 1153 | 28.8% | 954/1153 |

Not sparse: degree-2 needs 954 of its 1153 terms, and still only explains 29% of the variance. In hindsight this is the expected result, not a surprising one — `n_qubits=3` with a `ring` entangler is **already a complete graph** (every qubit reaches every other qubit within one entangling round), so there's no qubit-connectivity locality to exploit at all, and a hardware-efficient ansatz is specifically designed to mix its parameters into a highly entangled state as efficiently as possible. Finding it *isn't* sparse is what a well-designed entangling ansatz should look like.

`lagh.fit` (the unbounded scout, cheapest tool available) also timed out past 300s on this same 24-D data — independent confirmation that this is outside the practically computable range for both a general symbolic search and a naive dense/sparse regression.

## Tested and largely refuted: does a wider, sparser-topology register help?

The natural follow-up: `n_qubits=3` with `ring` is a complete graph, so of course there's no locality to exploit — what if the register were wider with a genuinely sparse topology (`chain`, at width ≥10)? Tested directly, and the headline answer is **no, not usefully**:

- **Pairwise interaction does shrink with qubit distance, but doesn't vanish.** Fitting the exact 9-term basis (residuals confirmed at machine precision) to two knobs on the same feature, at `n_qubits=10`, `entangler="chain"`: an adjacent pair (`q0`,`q1`) has cross-term magnitude `0.356`; the maximally-separated pair (`q0`,`q9`, opposite ends of the chain) has `0.202` — smaller and missing its `sin` component entirely, but still substantial. `depth=1` and `depth=2` gave numerically identical results in every variant of this test, unexplained so far.
- **The aggregate number doesn't move.** Re-running the same degree-≤2 sparsity fit from the section above, but on the wide/`chain` config (`N=20` knobs: one `Ry`-layer-0 knob per qubit, both features): **R² = 29.3%**, against **28.8%** for the original narrow/`ring` config. Essentially no improvement from switching topology. The handful of comparatively larger coefficients at degree 2 are all of one coherent shape — `A.qN` paired with `B.qN`, the same qubit index compared *across* the two features' circuits — not the qubit-distance locality within one feature's circuit that this test was trying to exploit.

Two wrong intermediate results along the way, corrected here for the record: an early pass reported `cross_mag=0.000000` for every distance tested, which was a term-classification bug (checking whether a basis-function *name* contained the character `'1'`, when labels like `cos1`/`sin1` always contain one — the filter matched nothing, ever). A follow-up "brick-wall entangler" toy simulator, meant to confirm a fix, used the wrong separability test (additive, `o11−o10−o01+o00=0`, rather than the multiplicative test that's actually the right null hypothesis for quantum overlaps) and had its own bugs; it was dropped rather than debugged further, in favor of re-querying the already-trustworthy `q_orca` code directly with a corrected method.

**Verdict:** topology is not the lever. Register width and sparsity labels (`chain` vs `ring`) barely move the aggregate compressibility of the multi-knob space. Whatever is suppressing locality here is either specific to how `q_orca`'s entangler applies its gate pairs (sequentially within one nominal layer, not in parallel non-overlapping sub-steps — worth checking on its own terms, separately, before concluding this is fundamental) or a deeper property of hardware-efficient ansätze in general. Either way, this closes off the "narrow the general N-knob case via circuit design" direction; the per-case results in the sections above (single knob, shared slot, ≤3 knobs) remain the actionable scope.

## Verdict for `structural_floor()`

If a `structural_floor()`-like guarantee is ever added for HEA, it should be scoped to what's actually proven here, not left as a blanket promise:

- **Any single swept knob**, with everything else fixed: closed-form, provably `M ± √(A²+B²)` (reducing to `M ± |V|` in the degenerate case) — cheap to compute per-knob without any optimizer call, the same way `MPSRung1.structural_floor()` works today.
- **A shared slot across the target pair** (the existing `cluster_shared` `Cancellation` pattern, e.g. `knobs=["dogs.theta[0,0,0]", "birds.theta[0,0,0]"]`): also closed-form, same family.
- **A small number (≤3, maybe ≤5 with more samples) of independently-chosen knobs**: derivable case-by-case via the same circuit-theoretic argument plus an exact/near-exact curve fit — tractable, but not automatic, and not something to promise without doing the derivation.
- **The fully general per-feature multi-knob case** (what an unrestricted `Cancellation` optimizer over many knobs would actually explore): no defensible closed-form bound. The `NotImplementedError`'s current message ("deferred to a future research-track proposal") should probably be narrowed to say this explicitly, rather than leaving the door open to a general bound that this note shows doesn't exist in a useful form for this ansatz.

## Caveats

- All numeric coefficients above are specific to one dictionary configuration (`AnimalsHea`: `HEA_Rung2(depth=2)`, `entangler="ring"`, `rotations=("Ry","Rz")`, `n_qubits=3`, the `dog_poodle`/`bird_hawk` target pair with the `alpha`/`beta`/`gamma` values in [Reproducing the result](#reproducing-the-result)). Only the **functional-form and basis-size** claims (the `3^N` theorem, the entangler-round argument) are configuration-independent; the specific numbers are not.
- The N=24 non-sparsity result was first measured at `n_qubits=3` with `ring` (a complete graph already). [Tested and largely refuted](#tested-and-largely-refuted-does-a-wider-sparser-topology-register-help) directly: a wider, sparser-topology config (`n_qubits=10`, `chain`) gives essentially the same aggregate R² (29.3% vs 28.8% at degree ≤2) — topology is not the lever.
- `lagh` could not find *any* of the multi-variable results unaided — every certified or theory-derived multi-knob law here required a human-supplied hypothesis fed to `verify`, never an autonomous `recover`. That's a real, reportable gap in `lagh`'s own curriculum (no cross-column trig term for two distinct angular inputs), not a property of HEA.
- The same-feature 2-knob and 3-knob closed forms are **not** `lagh` certificates — they're proven by the circuit-theory argument and confirmed by an exact-precision curve fit (residuals at `1e-16`), but `lagh.verify` declined to pin the non-rational coefficients at any sigma tried. Don't cite them as certified; cite them as theory + confirmed fit.

## Reproducing the result

Requires `lagh` installed alongside this repo (`pip install -e ../lagh[mcp]`, or just the core library — `import lagh` needs no extra dependencies).

```python
from dataclasses import replace
import numpy as np
from polygram import Dictionary, Feature, HEA_Rung2

def build_dictionary() -> Dictionary:
    return Dictionary(
        name="AnimalsHea",
        features=[
            Feature("dog_poodle", "dogs", beta=-0.50, alpha=0.05, gamma=0.02),
            Feature("dog_beagle", "dogs", beta=-0.48, alpha=0.04, gamma=0.03),
            Feature("bird_hawk", "birds", beta=0.50, alpha=-0.04, gamma=0.02),
            Feature("bird_sparrow", "birds", beta=0.52, alpha=-0.03, gamma=0.01),
        ],
        hierarchy={"dogs": ["dog_poodle", "dog_beagle"],
                   "birds": ["bird_hawk", "bird_sparrow"]},
        encoding=HEA_Rung2(depth=2),
    )

base = build_dictionary()
a_idx, b_idx = base.feature_index("dog_poodle"), base.feature_index("bird_hawk")

# theta[r,d,q] indexes (rotation type, depth/layer, qubit) -- rotation r=0
# is "Ry" here since HEA_Rung2's default rotations=("Ry","Rz"). Single-knob
# sweep, e.g. theta[Ry, layer=0, qubit=0]:
def overlap(path, value):
    d = base.with_knob(path, value)
    return float(np.abs(d.gram()[a_idx, b_idx]) ** 2)

phis = np.linspace(0.0, np.pi, 40)
y = [overlap("bird_hawk.theta[0,0,0]", float(p)) for p in phis]

from lagh.mcp import core
print(core.recover([[float(p)] for p in phis], y, sigma=0.0))
```

The two-knob and 24-knob experiments follow the same `dictionary.with_knob(...)` /
`Feature(theta=...)` pattern, looping over a grid or random samples and
calling `core.recover` / `core.verify` / `core.fit` from `lagh.mcp.core`
on the resulting `(X, y)`.

## See also

- [`cancellation-phase-floor.md`](cancellation-phase-floor.md) — the `MPSRung1` `M ± |V|` floor this note extends to HEA.
- [`rung3-viability-spike.md`](rung3-viability-spike.md) — the sibling encoding's viability-spike methodology this note's decision structure borrows from.
- [`lagh`](https://github.com/jascal/lagh) — the certified law-discovery engine used throughout; see its own `docs/` for the certificate/abstention model and the `α ≤ |H|·q^h` significance bound.
