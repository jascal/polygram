"""`Regrower` — the post-compression regrow primitive.

Two-stage API:

    plan() -> RegrowPlan                            # cheap, runs k-means on residuals
    apply(plan, output_checkpoint=...)              # writes one new safetensors
        -> RegrowResult
    run(output_checkpoint=...) -> RegrowResult      # apply(plan(), ...)

`plan()` resolves the residual stream (either from cached residuals
or by running ONE GPT-2 forward pass per prompt), runs the named
strategy (currently `residual_kmeans`) on the residual stream, and
returns a deterministic `RegrowPlan`. No safetensors I/O.

`apply()` reads the source state-dict via
`safetensors.numpy.load_file`, applies the strategy's tensor
population in-memory, writes the rewritten state atomically (sibling
temp + `os.replace`), and rebuilds a `Dictionary` from the new
checkpoint. Source bytes are never modified.

`run()` is the convenience wrapper.

Two construction modes:

- Direct: `Regrower(sae_checkpoint=..., strategy=..., zeroed=...,
  ...)`. The orca-lang demo path; the isolation-test path.
- Chained: `Regrower.from_compression_report(report, sae_checkpoint,
  ..., strategy=...)`. Extracts `zeroed` from the report's
  clusters; populates `RegrowReport.provenance` with the upstream
  report's identifying hashes.

The `Regrower` is torch-free. The `prompts` construction path
triggers a lazy torch + transformers import inside
`_capture_residuals`. The `cached_residuals` path is fully
torch-free.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from polygram.config import RegrowConfig  # noqa: F401

import numpy as np

from polygram.compression._hash import sha256_file
from polygram.compression.regrow_report import (
    SCHEMA_VERSION,
    RegrowPlan,
    RegrowReport,
    RegrowResult,
    SlotPopulation,
)


class RegrowStrategy(str, Enum):
    RESIDUAL_KMEANS = "residual_kmeans"
    HIGH_DECODER_NORM_RANDOM = "high_decoder_norm_random"
    ORTHOGONAL_NOISE_SCALED = "orthogonal_noise_scaled"


_SUPPORTED_STRATEGIES: frozenset[str] = frozenset(s.value for s in RegrowStrategy)
_IMPLEMENTED_STRATEGIES: frozenset[str] = frozenset({
    RegrowStrategy.RESIDUAL_KMEANS.value,
})


@dataclass
class Regrower:
    """Repopulates zeroed SAE slots with new directions.

    `strategy` is required (no default). Currently only
    `"residual_kmeans"` is implemented; the other enum members are
    reserved for follow-up changes and raise `NotImplementedError`
    when invoked.

    Exactly one of `prompts` or `cached_residuals` must be supplied.
    The `prompts` path triggers a lazy torch import to capture the
    residual stream from a GPT-2 forward pass; the
    `cached_residuals` path is fully torch-free.

    `from_compression_report` is the chained constructor: it
    extracts `zeroed` from the report's clusters and populates
    `RegrowReport.provenance` with the upstream report's identifying
    hashes. The direct constructor leaves provenance empty.
    """

    sae_checkpoint: Path
    strategy: str
    zeroed: set[int]
    seed: int = 0
    n_init: int = 4
    prompts: Sequence[str] | None = None
    cached_residuals: np.ndarray | None = None
    model_name: str = "gpt2"
    layer: int = 10
    device: str | None = None
    top_k: int | None = None

    # Cached plan + provenance, populated lazily.
    _cached_plan: RegrowPlan | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _provenance: dict[str, str] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    # ----------------------------------------------------------------
    # Construction-time validation
    # ----------------------------------------------------------------

    def __post_init__(self) -> None:
        self.sae_checkpoint = Path(self.sae_checkpoint)
        if not self.sae_checkpoint.is_file():
            raise ValueError(
                f"Regrower: sae_checkpoint not found: {self.sae_checkpoint}"
            )
        if self.strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Regrower: unsupported strategy {self.strategy!r}; "
                f"supported: {sorted(_SUPPORTED_STRATEGIES)}"
            )
        if int(self.seed) < 0:
            raise ValueError(
                f"Regrower: seed must be >= 0; got {self.seed}"
            )
        if int(self.n_init) < 1:
            raise ValueError(
                f"Regrower: n_init must be >= 1; got {self.n_init}"
            )
        if self.top_k is not None and int(self.top_k) < 0:
            raise ValueError(
                f"Regrower: top_k must be None or a non-negative int; "
                f"got top_k={self.top_k}"
            )

        # XOR on residual source
        if (self.prompts is None) == (self.cached_residuals is None):
            raise ValueError(
                "Regrower: exactly one of `prompts` or `cached_residuals` "
                "must be supplied; supplying neither or both is rejected"
            )

        if self.prompts is not None:
            if not list(self.prompts):
                raise ValueError("Regrower: `prompts` must be non-empty")
            if int(self.layer) < 0:
                raise ValueError(
                    f"Regrower: layer must be >= 0; got {self.layer}"
                )

        if self.cached_residuals is not None:
            arr = np.asarray(self.cached_residuals)
            if arr.ndim != 2:
                raise ValueError(
                    f"Regrower: cached_residuals must be 2D "
                    f"(n_tokens, d_model); got shape {arr.shape!r}"
                )
            if arr.dtype not in (np.float32, np.float64):
                raise ValueError(
                    f"Regrower: cached_residuals dtype must be float32 or "
                    f"float64; got {arr.dtype!r}"
                )
            object.__setattr__(self, "cached_residuals", arr.astype(
                np.float32, copy=False
            ))

        # Validate zeroed: non-negative ints, all within [0, n_features).
        if not isinstance(self.zeroed, set):
            self.zeroed = set(self.zeroed)
        for fid in self.zeroed:
            if not isinstance(fid, (int, np.integer)) or int(fid) < 0:
                raise ValueError(
                    f"Regrower: every fid in zeroed must be a non-negative "
                    f"int; got {fid!r}"
                )
        # Reading the safetensors header is cheap and avoids loading
        # all weights just to check the shape.
        n_features = _read_n_features(self.sae_checkpoint)
        for fid in self.zeroed:
            if int(fid) >= n_features:
                raise ValueError(
                    f"Regrower: zeroed contains fid {int(fid)} which is "
                    f"out of range for the source checkpoint's "
                    f"n_features={n_features}"
                )

    # ----------------------------------------------------------------
    # Chained constructor
    # ----------------------------------------------------------------

    @classmethod
    def from_compression_report(
        cls,
        report,  # CompressionReport — quoted to keep the import lazy-free
        sae_checkpoint: str | os.PathLike,
        *,
        strategy: str | None = None,
        prompts: Sequence[str] | None = None,
        cached_residuals: np.ndarray | None = None,
        seed: int | None = None,
        n_init: int | None = None,
        model_name: str | None = None,
        layer: int | None = None,
        device: str | None = None,
        top_k: int | None = None,
        config: "RegrowConfig | None" = None,
    ) -> "Regrower":
        # Precedence: per-field kwarg (non-None) > config > error for
        # required fields (model_name, layer have no default — silently
        # falling back to GPT-2 layer 10 was the pre-change footgun).
        if config is not None:
            if strategy is None:
                strategy = config.strategy
            if prompts is None and config.prompts is not None:
                prompts = list(config.prompts)
            if seed is None:
                seed = config.seed
            if n_init is None:
                n_init = config.n_init
            if model_name is None:
                model_name = config.model_name
            if layer is None:
                layer = config.layer
            if device is None:
                device = config.device
            if top_k is None:
                top_k = config.top_k
        # Required-field enforcement: when no config supplies them and no
        # per-field kwarg is given, raise a clear TypeError. The previous
        # ``model_name="gpt2"`` and ``layer=10`` defaults silently bound
        # the regrower to a GPT-2-shaped host model — incorrect for any
        # other architecture.
        missing = [
            name
            for name, value in (("model_name", model_name), ("layer", layer))
            if value is None
        ]
        if missing:
            raise TypeError(
                f"Regrower.from_compression_report missing required keyword "
                f"argument(s): {', '.join(missing)}. Pass them explicitly or "
                f"via config=RegrowConfig(model_name=..., layer=...)."
            )
        # Fill defaults for the optional fields we made sentinels above.
        if strategy is None:
            strategy = "residual_kmeans"
        if seed is None:
            seed = 0
        if n_init is None:
            n_init = 4
        zeroed: set[int] = {
            int(fid)
            for cluster in report.plan.clusters
            for fid in cluster.zeroed
        }
        instance = cls(
            sae_checkpoint=Path(sae_checkpoint),
            strategy=strategy,
            zeroed=zeroed,
            seed=seed,
            n_init=n_init,
            prompts=prompts,
            cached_residuals=cached_residuals,
            model_name=model_name,
            layer=layer,
            device=device,
            top_k=top_k,
        )
        object.__setattr__(
            instance,
            "_provenance",
            {
                "compression_report_source_sha256":
                    str(report.source_checkpoint_sha256),
                "compression_report_output_sha256":
                    str(report.output_checkpoint_sha256),
                "compression_report_dictionary_name":
                    str(report.validation_report_dictionary_name),
            },
        )
        return instance

    # ----------------------------------------------------------------
    # plan()
    # ----------------------------------------------------------------

    def plan(self) -> RegrowPlan:
        """Build the regrowth plan: resolve residuals, run the
        strategy, return a `RegrowPlan` with per-slot diagnostics.
        """
        if self._cached_plan is not None:
            return self._cached_plan

        from safetensors.numpy import load_file

        state_dict = load_file(str(self.sae_checkpoint))
        feature_ids = tuple(range(int(state_dict["W_dec"].shape[0])))

        residuals = self._resolve_residuals()
        zeroed_sorted = sorted(int(f) for f in self.zeroed)

        # top_k cap: regrow only the first top_k zeroed slots in plan
        # order. None (default) preserves byte-equivalence with the
        # pre-change behavior; a value >= len(zeroed_sorted) is a no-op.
        if self.top_k is not None and self.top_k < len(zeroed_sorted):
            zeroed_sorted = zeroed_sorted[: self.top_k]

        if not zeroed_sorted:
            # Empty-zeroed-set is a no-op: empty plan.
            plan = RegrowPlan(
                strategy=self.strategy,
                n_residual_tokens=int(residuals.shape[0]),
                zeroed_input=tuple(),
                feature_ids=feature_ids,
                slots=tuple(),
            )
            object.__setattr__(self, "_cached_plan", plan)
            return plan

        slots = self._dispatch_plan(state_dict, residuals, zeroed_sorted)

        plan = RegrowPlan(
            strategy=self.strategy,
            n_residual_tokens=int(residuals.shape[0]),
            zeroed_input=tuple(zeroed_sorted),
            feature_ids=feature_ids,
            slots=tuple(slots),
        )
        # Stash the materialized state-dict + slots so apply() can
        # reuse them without re-running k-means.
        object.__setattr__(self, "_cached_plan", plan)
        object.__setattr__(self, "_cached_residuals_array", residuals)
        return plan

    def _resolve_residuals(self) -> np.ndarray:
        if self.cached_residuals is not None:
            return self.cached_residuals
        return _capture_residuals(
            list(self.prompts),
            model_name=self.model_name,
            layer=int(self.layer),
            device=self.device,
        )

    def _dispatch_plan(
        self,
        state_dict: dict[str, np.ndarray],
        residuals: np.ndarray,
        zeroed_sorted: list[int],
    ) -> list[SlotPopulation]:
        if self.strategy == RegrowStrategy.RESIDUAL_KMEANS.value:
            from polygram.compression.strategies.residual_kmeans import (
                compute_residual_stream,
                plan_kmeans,
                apply_residual_kmeans,
            )

            residual_stream = compute_residual_stream(state_dict, residuals)
            centroids, cluster_sizes = plan_kmeans(
                residual_stream,
                zeroed_sorted,
                seed=int(self.seed),
                n_init=int(self.n_init),
            )
            _rewritten, slots = apply_residual_kmeans(
                state_dict, zeroed_sorted, centroids, cluster_sizes
            )
            # Stash the rewritten state-dict so apply() can reuse it.
            object.__setattr__(self, "_cached_rewritten_state", _rewritten)
            return slots

        # Reserved strategies — bodies not yet implemented.
        raise NotImplementedError(
            f"Regrower: strategy {self.strategy!r} is reserved for a future "
            f"change; supply 'residual_kmeans' for now"
        )

    # ----------------------------------------------------------------
    # apply()
    # ----------------------------------------------------------------

    def apply(
        self,
        plan: RegrowPlan | None = None,
        output_checkpoint: str | os.PathLike | None = None,
    ) -> RegrowResult:
        if output_checkpoint is None:
            raise ValueError(
                "Regrower.apply: output_checkpoint is required"
            )
        if plan is None:
            plan = self.plan()

        out_path = Path(output_checkpoint).resolve()
        if out_path == self.sae_checkpoint.resolve():
            raise ValueError(
                f"Regrower.apply: output_checkpoint must differ from the "
                f"source checkpoint (both resolved to {out_path})"
            )

        from safetensors.numpy import load_file, save_file

        from polygram.sae_import import from_sae_lens, load_sae_safetensors

        # Reuse the cached rewritten state from plan() if available;
        # otherwise re-materialize.
        rewritten = getattr(self, "_cached_rewritten_state", None)
        if rewritten is None:
            # Empty zeroed-set path: rewritten == source.
            rewritten = load_file(str(self.sae_checkpoint))

        source_sha = sha256_file(self.sae_checkpoint)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(out_path.parent),
            prefix=f".{out_path.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_path = Path(tmp.name)
        try:
            save_file(rewritten, str(tmp_path))
            os.replace(tmp_path, out_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

        output_sha = sha256_file(out_path)

        n_repopulated = sum(1 for s in plan.slots if s.cluster_size > 0)
        n_left_zero = sum(1 for s in plan.slots if s.cluster_size == 0)

        report = RegrowReport(
            schema_version=SCHEMA_VERSION,
            source_checkpoint=str(self.sae_checkpoint),
            source_checkpoint_sha256=source_sha,
            output_checkpoint=str(out_path),
            output_checkpoint_sha256=output_sha,
            strategy=self.strategy,
            plan=plan,
            n_slots_repopulated=n_repopulated,
            n_slots_left_zero=n_left_zero,
            strategy_params={
                "seed": int(self.seed),
                "n_init": int(self.n_init),
            },
            provenance=dict(self._provenance),
        )

        # Pick a feature subset for the Dictionary rebuild — MPSRung1
        # caps Dictionary at `MPSRung1.max_features` (= 8), so we
        # can't rebuild on the full 24K-feature SAE. Prefer the
        # populated slots (most interesting for inspection); fall back
        # to the first `cap` feature ids when there are no populated
        # slots.
        # When Regrower gains a configurable encoding (issue #48),
        # this should query the supplied encoding's max_features
        # rather than hardcoding MPSRung1.
        from polygram.encoding import MPSRung1 as _MPSRung1

        rebuild_cap = int(_MPSRung1.max_features)
        sorted_zeroed = sorted(int(f) for f in self.zeroed)
        if sorted_zeroed:
            dict_ids = sorted_zeroed[:rebuild_cap]
        else:
            dict_ids = list(
                plan.feature_ids[: min(rebuild_cap, len(plan.feature_ids))]
            )
        records = load_sae_safetensors(str(out_path), feature_ids=dict_ids)
        rebuilt_dictionary, _ = from_sae_lens(
            records,
            dict_ids,
            assign_gamma=True,
            name=self._provenance.get(
                "compression_report_dictionary_name",
                f"Regrown_{self.sae_checkpoint.stem.replace('-', '_').replace('.', '_')}",
            ),
        )

        return RegrowResult(
            plan=plan,
            report=report,
            output_checkpoint=out_path,
            dictionary=rebuilt_dictionary,
        )

    # ----------------------------------------------------------------
    # run()
    # ----------------------------------------------------------------

    def run(
        self, output_checkpoint: str | os.PathLike
    ) -> RegrowResult:
        return self.apply(self.plan(), output_checkpoint=output_checkpoint)


# ============================================================================
# Helpers
# ============================================================================


def _read_n_features(path: Path) -> int:
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        if "W_dec" in f.keys():
            return int(f.get_slice("W_dec").get_shape()[0])
        # Fallback: probe the standard precedence list.
        for cand in ("decoder.weight", "dec"):
            if cand in f.keys():
                shape = f.get_slice(cand).get_shape()
                # decoder.weight in PyTorch convention is (out, in) = (d_model, d_sae)
                # so n_features is shape[1] for that layout.
                if cand == "decoder.weight" and shape[0] != shape[1]:
                    return int(shape[1])
                return int(shape[0])
        raise KeyError(
            f"_read_n_features: no decoder key in {path}; expected one of "
            f"W_dec, decoder.weight, dec"
        )


def _capture_residuals(
    prompts: list[str],
    *,
    model_name: str,
    layer: int,
    device: str | None,
) -> np.ndarray:
    """Lazy-import torch + transformers; run one forward per prompt
    with a pre-hook at `model.transformer.h[layer]`; return the
    concatenated residuals as float32."""
    from polygram.behavioural.runtime import (
        _get_layer_module,
        _import_torch_and_transformers,
        _resolve_device,
    )

    torch, AutoModelForCausalLM, AutoTokenizer = _import_torch_and_transformers()
    resolved = _resolve_device(torch, device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(resolved)

    captured: list[np.ndarray] = []

    def _hook(module, args):
        captured.append(args[0].detach().cpu().numpy())

    handle = _get_layer_module(model, int(layer)).register_forward_pre_hook(_hook)
    chunks: list[np.ndarray] = []
    try:
        for prompt in prompts:
            captured.clear()
            toks = tokenizer(prompt, return_tensors="pt")
            toks = {k: v.to(resolved) for k, v in toks.items()}
            with torch.no_grad():
                model(**toks)
            chunks.append(captured[0][0].astype(np.float32))
    finally:
        handle.remove()
    return np.concatenate(chunks, axis=0)
