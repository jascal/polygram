"""Behavioural-validator runtime helpers.

Lazy imports of `torch` + `transformers` are kept behind
`_import_torch_and_transformers()` so that `polygram.behavioural` is
importable without those extras (matches the §4.4 spike pattern).
`predict()` never crosses this boundary; only `validate()` does.
"""

from __future__ import annotations

import math
import warnings

import numpy as np


_BEHAVIOURAL_INSTALL_HINT = (
    "torch + transformers are required for "
    "BehaviouralValidator.validate(); "
    "install via `pip install polygram[behavioural]`."
)


def _resolve_device(torch_module, requested: str | None) -> str:
    """Resolve a device preference to a concrete device string.

    `requested` is one of None / "auto" / "cuda" / "mps" / "cpu". `None`
    and `"auto"` pick the best available accelerator: cuda → mps → cpu.
    Explicit `"cuda"` / `"mps"` requests raise `ValueError` if the
    backend isn't usable on this machine. Falling back to CPU under
    auto-resolution emits a `RuntimeWarning` so callers running on
    larger-than-GPT-2 models get a heads-up that the run will be slow.

    Returns the resolved device string.
    """
    norm = (requested or "auto").lower()
    if norm not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(
            f"_resolve_device: unsupported device {requested!r}; "
            f"expected one of auto / cuda / mps / cpu"
        )

    cuda_ok = bool(getattr(torch_module, "cuda", None)) and bool(
        getattr(torch_module.cuda, "is_available", lambda: False)()
    )
    mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
    mps_ok = bool(mps_backend) and bool(
        getattr(mps_backend, "is_available", lambda: False)()
    )

    if norm == "cuda":
        if not cuda_ok:
            raise ValueError(
                "_resolve_device: device='cuda' requested but no CUDA "
                "device is available on this machine"
            )
        return "cuda"
    if norm == "mps":
        if not mps_ok:
            raise ValueError(
                "_resolve_device: device='mps' requested but the MPS "
                "backend is not available on this machine (requires "
                "Apple Silicon + a torch build with MPS support)"
            )
        return "mps"
    if norm == "cpu":
        return "cpu"

    # auto
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    warnings.warn(
        "BehaviouralValidator: no GPU backend available — running on "
        "CPU. Validator runs scale roughly with `n_features × n_prompts`; "
        "GPT-2 small finishes in ~10–15 min on CPU but larger models "
        "(Gemma, Llama, etc.) may take hours. Pass `device='cuda'`/"
        "`'mps'` explicitly to surface backend availability errors "
        "instead of silently falling back.",
        RuntimeWarning,
        stacklevel=2,
    )
    return "cpu"


def _import_torch_and_transformers():
    """Lazy-import torch + transformers; raise `ImportError` with a
    pip-install hint when either is missing.

    Returns `(torch_module, AutoModelForCausalLM, AutoTokenizer)`.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401
    except ImportError as exc:
        raise ImportError(_BEHAVIOURAL_INSTALL_HINT) from exc
    import torch as _torch
    from transformers import AutoModelForCausalLM as _AutoModelForCausalLM
    from transformers import AutoTokenizer as _AutoTokenizer

    return _torch, _AutoModelForCausalLM, _AutoTokenizer


def _get_layer_module(model, layer: int):
    """Return the transformer block at `layer` for the given model.

    Handles GPT-2 family (`model.transformer.h`) and
    Llama / Gemma / Mistral family (`model.model.layers`). Raises
    `ValueError` for unrecognised architectures.
    """
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[layer]
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[layer]
    arch = type(model).__name__
    raise ValueError(
        f"_get_layer_module: unsupported model architecture {arch!r}; "
        f"expected a GPT-2-family model (model.transformer.h) or a "
        f"Llama/Gemma-family model (model.model.layers). "
        f"Override _get_layer_module or file a bug to add support."
    )


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
