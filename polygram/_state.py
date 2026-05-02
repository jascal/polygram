"""Local 3-qubit statevector simulation for the rung-1 MPS staircase.

We re-implement the encoder here (rather than depending on q-orca's
private `_build_concept_state`) so Polygram can compute per-feature
quantities (Schmidt rank, entanglement entropy in future) without
poking through q-orca's internals.
"""

from __future__ import annotations

import numpy as np

from polygram.dictionary import Feature

N_QUBITS = 3


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(theta: float) -> np.ndarray:
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(complex)


def _apply_1q(state: np.ndarray, u: np.ndarray, qubit: int) -> np.ndarray:
    t = state.reshape((2,) * N_QUBITS)
    t = np.tensordot(u, t, axes=([1], [qubit]))
    t = np.moveaxis(t, 0, qubit)
    return t.reshape(2**N_QUBITS)


def _apply_cnot(state: np.ndarray, control: int, target: int) -> np.ndarray:
    t = state.reshape((2,) * N_QUBITS).copy()
    idx: list = [slice(None)] * N_QUBITS
    idx[control] = 1
    block = t[tuple(idx)]
    target_axis = target if target < control else target - 1
    t[tuple(idx)] = np.flip(block, axis=target_axis)
    return t.reshape(2**N_QUBITS)


def build_statevector(feature: Feature) -> np.ndarray:
    """Apply the cross-coupled rung-1 staircase to |000>."""
    state = np.zeros(2**N_QUBITS, dtype=complex)
    state[0] = 1.0
    state = _apply_1q(state, _ry(feature.alpha), 0)
    state = _apply_cnot(state, 0, 1)
    state = _apply_1q(state, _ry(feature.alpha + feature.beta), 1)
    state = _apply_1q(state, _rz(feature.phi), 1)
    state = _apply_cnot(state, 1, 2)
    state = _apply_1q(state, _ry(feature.beta + feature.gamma), 2)
    return state


def schmidt_rank(state: np.ndarray, cut: int = 1, tol: float = 1e-9) -> int:
    """Schmidt rank at bipartition `(q0..q{cut-1}) | (q{cut}..q{N-1})`."""
    m = state.reshape(2**cut, 2 ** (N_QUBITS - cut))
    sv = np.linalg.svd(m, compute_uv=False)
    return int(np.sum(sv > tol))
