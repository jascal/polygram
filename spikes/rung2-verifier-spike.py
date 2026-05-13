"""Rung-2 HEA verifier spike — answers: can simulation-based tier-ordering
verification work at depth=5, n=8?

Throwaway. Numpy only — no QuTiP, no q-orca, no polygram. Run as:

    python spikes/rung2-verifier-spike.py

Probes three questions:

  1. Runtime cost of full Gram via dense statevector across (n_qubits, depth).
  2. Tier-ordering verification pass-rate as a function of intra-cluster
     parameter jitter and tolerance — gives us the tolerance window the
     verifier should adopt.
  3. Phase-knob reconnaissance: under HEA, does varying a single rotation
     parameter still factor as M + V·cos(δ)? (Determines whether the
     rung-1 structural_floor diagnostic generalizes.)
"""

from __future__ import annotations

import time

import numpy as np


# ---------------------------------------------------------------------------
# Single-qubit and two-qubit gates
# ---------------------------------------------------------------------------


def Rx(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def Ry(t: float) -> np.ndarray:
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def Rz(t: float) -> np.ndarray:
    return np.array(
        [[np.exp(-1j * t / 2), 0.0], [0.0, np.exp(1j * t / 2)]],
        dtype=complex,
    )


def apply_single(
    state: np.ndarray, U: np.ndarray, qubit: int, n_qubits: int
) -> np.ndarray:
    """Apply 2x2 gate U to `qubit` of an `n_qubits` statevector."""
    s = state.reshape((2,) * n_qubits)
    s = np.tensordot(U, s, axes=([1], [qubit]))
    s = np.moveaxis(s, 0, qubit)
    return s.reshape(2**n_qubits)


def apply_cnot(
    state: np.ndarray, ctrl: int, tgt: int, n_qubits: int
) -> np.ndarray:
    s = state.reshape((2,) * n_qubits).copy()
    sel: list = [slice(None)] * n_qubits
    sel[ctrl] = 1
    sub_axis = tgt if tgt < ctrl else tgt - 1
    s[tuple(sel)] = np.flip(s[tuple(sel)], axis=sub_axis)
    return s.reshape(2**n_qubits)


# ---------------------------------------------------------------------------
# HEA circuit
# ---------------------------------------------------------------------------


def hea_statevector(
    theta: np.ndarray,
    n_qubits: int,
    depth: int,
    entangler: str = "ring",
) -> np.ndarray:
    """theta shape: (3, depth, n_qubits) → Rx, Ry, Rz per layer per qubit."""
    if theta.shape != (3, depth, n_qubits):
        raise ValueError(f"theta shape {theta.shape} != (3, {depth}, {n_qubits})")
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    for layer in range(depth):
        for q in range(n_qubits):
            state = apply_single(state, Rx(theta[0, layer, q]), q, n_qubits)
        for q in range(n_qubits):
            state = apply_single(state, Ry(theta[1, layer, q]), q, n_qubits)
        for q in range(n_qubits):
            state = apply_single(state, Rz(theta[2, layer, q]), q, n_qubits)
        if entangler == "ring":
            for q in range(n_qubits):
                state = apply_cnot(state, q, (q + 1) % n_qubits, n_qubits)
        elif entangler == "chain":
            for q in range(n_qubits - 1):
                state = apply_cnot(state, q, q + 1, n_qubits)
        else:
            raise ValueError(f"unknown entangler {entangler!r}")
    return state


def compute_concept_gram_hea(
    thetas: list[np.ndarray],
    n_qubits: int,
    depth: int,
    entangler: str = "ring",
) -> tuple[np.ndarray, list[np.ndarray]]:
    states = [hea_statevector(t, n_qubits, depth, entangler) for t in thetas]
    n = len(states)
    g = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            g[i, j] = abs(np.vdot(states[i], states[j])) ** 2
    return g, states


# ---------------------------------------------------------------------------
# Synthetic clustered concepts (planted hierarchy)
# ---------------------------------------------------------------------------


def synth_concepts(
    n_clusters: int,
    per_cluster: int,
    jitter: float,
    seed: int,
    n_qubits: int,
    depth: int,
    center_scale: float = 1.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, center_scale, (n_clusters, 3, depth, n_qubits))
    thetas: list[np.ndarray] = []
    labels: list[int] = []
    for c, center in enumerate(centers):
        for _ in range(per_cluster):
            thetas.append(center + jitter * rng.normal(0.0, 1.0, center.shape))
            labels.append(c)
    return thetas, np.array(labels, dtype=int)


# ---------------------------------------------------------------------------
# Tier-ordering checks
# ---------------------------------------------------------------------------


def tier_stats(gram: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    n = len(labels)
    intra: list[float] = []
    cross: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            (intra if labels[i] == labels[j] else cross).append(float(gram[i, j]))
    out = {
        "intra_mean": float(np.mean(intra)) if intra else float("nan"),
        "cross_mean": float(np.mean(cross)) if cross else float("nan"),
    }
    if intra and cross:
        out["margin"] = out["intra_mean"] - out["cross_mean"]
    else:
        out["margin"] = float("nan")
    return out


def rung2_tier_ordering(
    gram: np.ndarray, labels: np.ndarray, tolerance: float = 0.0
) -> tuple[bool, int, int]:
    """For every (i, j) in same cluster and (i, k) in different cluster,
    require gram[i,j] + tolerance >= gram[i,k]. Returns
    (passed, violations, total_triples)."""
    n = len(labels)
    violations = 0
    total = 0
    for i in range(n):
        for j in range(n):
            if j == i or labels[j] != labels[i]:
                continue
            for k in range(n):
                if k == i or k == j or labels[k] == labels[i]:
                    continue
                total += 1
                if gram[i, j] + tolerance < gram[i, k]:
                    violations += 1
    return violations == 0, violations, total


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def runtime_sweep() -> list[dict]:
    print("\n=== Runtime sweep (8 concepts, ring entangler) ===")
    print(f"{'n_qubits':>9} {'depth':>6} {'gram_s':>9} {'states_s':>10}")
    rows: list[dict] = []
    for n_q in [6, 8, 10]:
        for d in [3, 5, 7]:
            thetas, _ = synth_concepts(
                n_clusters=4,
                per_cluster=2,
                jitter=0.1,
                seed=0,
                n_qubits=n_q,
                depth=d,
            )
            t0 = time.perf_counter()
            states = [hea_statevector(t, n_q, d, "ring") for t in thetas]
            t_states = time.perf_counter() - t0
            t0 = time.perf_counter()
            _g, _ = compute_concept_gram_hea(thetas, n_q, d, "ring")
            t_gram = time.perf_counter() - t0
            print(f"{n_q:>9} {d:>6} {t_gram:>9.3f} {t_states:>10.3f}")
            rows.append(
                {
                    "n_qubits": n_q,
                    "depth": d,
                    "gram_s": t_gram,
                    "states_s": t_states,
                }
            )
            del states
    return rows


def tier_ordering_sweep(n_seeds: int = 16) -> list[dict]:
    print(
        f"\n=== Tier-ordering sweep "
        f"(n=8, depth=5, ring, 4 clusters x 2, {n_seeds} seeds) ==="
    )
    print(
        f"{'jitter':>7} {'tol':>6} {'pass_rate':>10} "
        f"{'mean_viol':>10} {'intra':>8} {'cross':>8} {'margin':>8}"
    )
    rows: list[dict] = []
    for jitter in [0.05, 0.10, 0.25, 0.50]:
        for tol in [0.0, 0.025, 0.05, 0.10]:
            passes = 0
            viols: list[int] = []
            intras: list[float] = []
            crosses: list[float] = []
            margins: list[float] = []
            for seed in range(n_seeds):
                thetas, labels = synth_concepts(
                    n_clusters=4,
                    per_cluster=2,
                    jitter=jitter,
                    seed=seed,
                    n_qubits=8,
                    depth=5,
                )
                g, _ = compute_concept_gram_hea(thetas, 8, 5, "ring")
                ok, v, _ = rung2_tier_ordering(g, labels, tolerance=tol)
                passes += int(ok)
                viols.append(v)
                stats = tier_stats(g, labels)
                intras.append(stats["intra_mean"])
                crosses.append(stats["cross_mean"])
                margins.append(stats["margin"])
            print(
                f"{jitter:>7} {tol:>6} {passes / n_seeds:>10.2f} "
                f"{np.mean(viols):>10.2f} {np.mean(intras):>8.4f} "
                f"{np.mean(crosses):>8.4f} {np.mean(margins):>8.4f}"
            )
            rows.append(
                {
                    "jitter": jitter,
                    "tolerance": tol,
                    "pass_rate": passes / n_seeds,
                    "mean_violations": float(np.mean(viols)),
                    "intra_mean": float(np.mean(intras)),
                    "cross_mean": float(np.mean(crosses)),
                    "margin": float(np.mean(margins)),
                }
            )
    return rows


def cancellation_floor_recon() -> dict:
    """Probe whether |<A|B>|^2 vs a single HEA knob still factors as
    M + V·cos(δ). FFT spectrum of the overlap curve over [0, 2π] tells us:
    purely sinusoidal → only c1 nonzero. Otherwise the rung-1 diagnostic
    needs redefinition."""
    print("\n=== Phase-knob reconnaissance: vary one HEA param of feature A ===")
    rng = np.random.default_rng(0)
    n_q, depth = 8, 5
    theta_A = rng.normal(0, 1.0, (3, depth, n_q))
    theta_B = theta_A + 0.10 * rng.normal(0, 1.0, (3, depth, n_q))

    knobs = [
        ("Rz q0 layer-0", (2, 0, 0)),
        ("Rz q0 layer-4", (2, 4, 0)),
        ("Ry q3 layer-2", (1, 2, 3)),
    ]
    out: dict = {}
    grid = np.linspace(0.0, 2 * np.pi, 33)
    for name, axis in knobs:
        overlaps = []
        for v in grid:
            ta = theta_A.copy()
            ta[axis] = v
            sa = hea_statevector(ta, n_q, depth, "ring")
            sb = hea_statevector(theta_B, n_q, depth, "ring")
            overlaps.append(abs(np.vdot(sa, sb)) ** 2)
        overlaps_arr = np.array(overlaps)
        omin = float(overlaps_arr.min())
        omax = float(overlaps_arr.max())
        # FFT on the periodic samples (drop the wrap point)
        coeffs = np.fft.rfft(overlaps_arr[:-1]) / (len(overlaps_arr) - 1)
        amps = np.abs(coeffs)
        # purity = c1 / sum(c1..c_max); higher → more sinusoidal
        higher = amps[1:].sum()
        purity = float(amps[1] / higher) if higher > 0 else float("nan")
        print(
            f"  {name:>16}: min={omin:.4f} max={omax:.4f} "
            f"swing={omax-omin:.4f} c0..c4={amps[:5].round(4).tolist()} "
            f"purity(c1/Σ)={purity:.3f}"
        )
        out[name] = {
            "min": omin,
            "max": omax,
            "swing": omax - omin,
            "amps": amps[:5].tolist(),
            "purity": purity,
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("Rung-2 HEA verifier spike")
    print("=" * 60)
    runtime_rows = runtime_sweep()
    tier_rows = tier_ordering_sweep(n_seeds=16)
    floor_rows = cancellation_floor_recon()

    print("\n=== Summary ===")
    n8d5 = next(r for r in runtime_rows if r["n_qubits"] == 8 and r["depth"] == 5)
    print(f"  Reference cell (n=8, depth=5): Gram took {n8d5['gram_s']:.3f}s")
    full_pass = [r for r in tier_rows if r["pass_rate"] == 1.0]
    if full_pass:
        cheapest = min(full_pass, key=lambda r: r["tolerance"])
        print(
            f"  Lowest tolerance with 100% pass at n=8 d=5: "
            f"jitter={cheapest['jitter']} tol={cheapest['tolerance']}"
        )
    else:
        print("  No (jitter, tol) cell hit 100% pass — see table above")
    purities = [v["purity"] for v in floor_rows.values()]
    print(
        f"  Phase-knob purity (c1/Σ) across 3 knobs: "
        f"min={min(purities):.3f} max={max(purities):.3f} — "
        f"{'sinusoidal' if min(purities) > 0.9 else 'NOT pure cos(δ)'}"
    )


if __name__ == "__main__":
    main()
