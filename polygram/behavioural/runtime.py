"""Behavioural-validator runtime helpers.

Lazy imports of `torch` + `transformers` are kept behind
`_import_torch_and_transformers()` so that `polygram.behavioural` is
importable without those extras (matches the §4.4 spike pattern).
`predict()` never crosses this boundary; only `validate()` does.
"""

from __future__ import annotations

import math

import numpy as np


_BEHAVIOURAL_INSTALL_HINT = (
    "torch + transformers are required for "
    "BehaviouralValidator.validate(); "
    "install via `pip install polygram[behavioural]`."
)


def _import_torch_and_transformers():
    """Lazy-import torch + transformers; raise `ImportError` with a
    pip-install hint when either is missing.

    Returns `(torch_module, GPT2LMHeadModel, GPT2Tokenizer)`.
    """
    try:
        import torch  # noqa: F401
        from transformers import GPT2LMHeadModel  # noqa: F401
        from transformers import GPT2Tokenizer  # noqa: F401
    except ImportError as exc:
        raise ImportError(_BEHAVIOURAL_INSTALL_HINT) from exc
    import torch as _torch
    from transformers import GPT2LMHeadModel as _GPT2LMHeadModel
    from transformers import GPT2Tokenizer as _GPT2Tokenizer

    return _torch, _GPT2LMHeadModel, _GPT2Tokenizer


def _kl_softmax_row(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """Per-token KL between two next-token logit rows. Returns max(0, KL)
    so float32 noise on near-identical distributions can't drive the
    algebraic non-negativity below zero."""
    log_p = logits_a - float(np.logaddexp.reduce(logits_a, axis=-1))
    log_q = logits_b - float(np.logaddexp.reduce(logits_b, axis=-1))
    p = np.exp(log_p)
    return float(max(0.0, np.sum(p * (log_p - log_q))))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _bootstrap_ci_mean(
    values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05
) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(0)
    means = np.array(
        [
            rng.choice(values, size=values.size, replace=True).mean()
            for _ in range(n_resamples)
        ]
    )
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _safe_log_abs(ratio: float) -> float:
    if ratio is None or not math.isfinite(ratio) or ratio <= 0.0:
        return float("nan")
    return abs(math.log(ratio))
