# Theoretical treatments

Self-contained mathematical write-ups of the engineering primitives
shipped in this codebase. These are research-track artefacts —
the algorithmic invariants they prove are the same ones our test
suite exercises empirically, but the proofs and bounds live here
rather than in the spec.

## Documents

- **`polygram.pdf`** / **`polygram.tex`** — *Polygrams: A
  Theoretical Treatment of Kronecker-Succinct Vector Families*
  (May 2026). Formal study of the order-`n = M + k` Polygram
  (Kronecker product of single-qubit Bloch states). Establishes
  the factorised inner-product identity, manifold dimension /
  codimension, covering-number-based approximation lower bounds,
  Lipschitz stability, identifiability from `4n + 1` overlap
  measurements modulo the U(1)^{n-1} gauge, and the comparison
  to MPS / tensor trains (Polygrams = bond-dimension-1 MPS).
  Cited from
  [`docs/research/rung5-encoding.md`](../rung5-encoding.md) — the
  paper's `8 · 2^k` saturation result (Prop 4.1) is what the
  Rung5 rank-verification artifact confirms empirically for
  k ∈ {2, 3, 4}.

## Building the LaTeX

```
cd docs/research/theory
pdflatex polygram.tex && pdflatex polygram.tex
```

(Two passes to resolve the table-of-contents and cross-references.)
The committed `polygram.pdf` was built from this source; rebuild
when editing the `.tex`.
