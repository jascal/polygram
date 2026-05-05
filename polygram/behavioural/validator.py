"""BehaviouralValidator — the four-constraint loop pipeline.

Two-stage API:

    predict() -> list[CandidatePair]   # cheap, no torch
    validate(candidates=None)          # expensive, lazy torch import
        -> ValidationReport
    run() -> ValidationReport          # validate(predict())

`predict()` computes the predicted Polygram squared-overlap matrix and
the decoder-cosine² matrix and emits one `CandidatePair` per (i, j)
with `i < j`. Behavioural fields are NaN; `gate_pass` is `False`.

`validate()` lazy-imports torch + transformers, hooks
`model.transformer.h[layer]`, forwards every prompt, encodes the
captured residuals through the SAE, runs exactly N ablation forward-
pass-batches (one per feature), and aggregates per-pair statistics
from the cached per-token KL arrays. The cost cap (`MUST NOT run more
than n_features ablation batches`) is encoded as a contract per the
spec.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from polygram.behavioural.report import (
    BucketStats,
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)
from polygram.behavioural.runtime import (
    _bootstrap_ci_mean,
    _get_layer_module,
    _import_torch_and_transformers,
    _kl_softmax_row,
    _pearson,
    _resolve_device,
    _safe_log_abs,
    _spearman,
)
from polygram.dictionary import Dictionary
from polygram.sae_import import (
    MAX_FEATURES_PER_DICTIONARY,
    load_sae_safetensors,
)

SCHEMA_VERSION = 1


_LAYER_ZERO_MESSAGE = (
    "BehaviouralValidator: layer 0 is typically a structural dead zone "
    "with negligible KL signal per ablation (documented for GPT-2 small "
    "in docs/research/deeper-layer-ablation-probe.md; likely similar for "
    "other decoder-only architectures). Use a deeper layer, or pass "
    "allow_layer_zero=True if your model family has been empirically "
    "shown to be informative at layer 0."
)


@dataclass
class BehaviouralValidator:
    """Runs the four-constraint compression-loop pipeline against a
    `Dictionary` of SAE features.

    Defaults encode the §4.4 GPT-2-small calibration: the
    `polygram_overlap_threshold = 0.7` is the §4.4 high-bucket lower
    bound; `jaccard_threshold = 0.30` separates the §4.4 mid- and
    high-bucket CIs; `min_firing_rate = 0.01` matches §4.4's
    eligibility filter; `min_both_fire = 5` is the §4.4 KL-ratio
    definability gate.
    """

    dictionary: Dictionary
    sae_checkpoint: Path
    feature_ids: list[int]
    prompts: Sequence[str]
    layer: int
    model_name: str = "gpt2"
    polygram_overlap_threshold: float = 0.7
    jaccard_threshold: float = 0.30
    min_firing_rate: float = 0.01
    min_both_fire: int = 5
    allow_layer_zero: bool = False
    device: str | None = None

    # Internal cache of decoder rows after a successful predict() call.
    _decoder_rows_cache: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ----------------------------------------------------------------
    # Validation
    # ----------------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.feature_ids, list):
            self.feature_ids = list(self.feature_ids)
        n_dict = len(self.dictionary.features)
        n_ids = len(self.feature_ids)
        if n_ids != n_dict:
            raise ValueError(
                f"BehaviouralValidator.feature_ids: length {n_ids} "
                f"does not match dictionary.features length {n_dict}"
            )
        if n_ids > MAX_FEATURES_PER_DICTIONARY:
            raise ValueError(
                f"BehaviouralValidator.feature_ids: {n_ids} exceeds "
                f"MAX_FEATURES_PER_DICTIONARY="
                f"{MAX_FEATURES_PER_DICTIONARY} (rung-1 MPS encoding "
                f"cap from polygram/sae_import.py:23)"
            )

        for name, value in (
            ("polygram_overlap_threshold", self.polygram_overlap_threshold),
            ("jaccard_threshold", self.jaccard_threshold),
            ("min_firing_rate", self.min_firing_rate),
        ):
            fv = float(value)
            if not (0.0 <= fv <= 1.0):
                raise ValueError(
                    f"BehaviouralValidator.{name}: {fv} not in [0, 1]"
                )

        if int(self.min_both_fire) < 1:
            raise ValueError(
                f"BehaviouralValidator.min_both_fire: "
                f"{self.min_both_fire} must be >= 1"
            )

        if self.layer < 0:
            raise ValueError(
                f"BehaviouralValidator.layer: {self.layer} must be "
                f">= 0 (negative layers are never allowed; "
                f"allow_layer_zero only gates layer == 0)"
            )

        if self.layer == 0:
            if not self.allow_layer_zero:
                raise ValueError(_LAYER_ZERO_MESSAGE)
            warnings.warn(_LAYER_ZERO_MESSAGE, RuntimeWarning, stacklevel=2)

        if not self.prompts:
            raise ValueError(
                "BehaviouralValidator.prompts: empty sequence; "
                "supply at least one prompt"
            )

        sae_path = Path(self.sae_checkpoint)
        if not sae_path.is_file():
            raise ValueError(
                f"BehaviouralValidator.sae_checkpoint: file not found "
                f"on disk: {sae_path} (the validator does not "
                f"download)"
            )
        self.sae_checkpoint = sae_path

    # ----------------------------------------------------------------
    # predict() — Polygram-only stage (no torch)
    # ----------------------------------------------------------------

    def predict(self) -> list[CandidatePair]:
        """Compute Polygram's predicted squared-overlap matrix and the
        SAE-decoder cosine² matrix; emit one `CandidatePair` per
        `(i, j)` with `i < j`. Behavioural fields are NaN.

        SHALL NOT import torch or transformers.
        """
        gram = self.dictionary.gram()
        polygram_sq = np.abs(gram) ** 2

        records = load_sae_safetensors(
            self.sae_checkpoint, feature_ids=list(self.feature_ids)
        )
        decoder_rows = np.stack(
            [records[fid].projection for fid in self.feature_ids]
        )
        self._decoder_rows_cache = decoder_rows

        pairs: list[CandidatePair] = []
        n = len(self.feature_ids)
        for i_idx in range(n):
            wi = decoder_rows[i_idx]
            wi_norm_sq = float(np.dot(wi, wi))
            for j_idx in range(i_idx + 1, n):
                wj = decoder_rows[j_idx]
                wj_norm_sq = float(np.dot(wj, wj))
                denom = wi_norm_sq * wj_norm_sq
                if denom > 0:
                    decoder_overlap = float(np.dot(wi, wj)) ** 2 / denom
                else:
                    decoder_overlap = 0.0
                pairs.append(CandidatePair(
                    i=int(self.feature_ids[i_idx]),
                    j=int(self.feature_ids[j_idx]),
                    polygram_overlap=float(polygram_sq[i_idx, j_idx]),
                    decoder_overlap=float(decoder_overlap),
                    jaccard=float("nan"),
                    pearson_activation=float("nan"),
                    kl_ablate_i=float("nan"),
                    kl_ablate_j=float("nan"),
                    kl_ratio_paired=float("nan"),
                    kl_log_ratio_abs=float("nan"),
                    n_fires_i=0,
                    n_fires_j=0,
                    n_both_fire=0,
                    n_either_fire=0,
                    gate_pass=False,
                ))
        return pairs

    # ----------------------------------------------------------------
    # validate() — behavioural stage (lazy torch import)
    # ----------------------------------------------------------------

    def validate(
        self,
        candidates: list[CandidatePair] | None = None,
    ) -> ValidationReport:
        """Run the behavioural stage and return a `ValidationReport`.

        - Lazy-imports torch + transformers.
        - Hooks `model.transformer.h[layer]` (forward-pre-hook).
        - Forwards every prompt; captures residuals + baseline logits.
        - Encodes through the SAE for every selected feature.
        - Runs exactly `len(feature_ids)` ablation forward-pass-batches
          (one per feature). The validator MUST NOT run a separate
          forward pass per pair.
        """
        if candidates is None:
            candidates = self.predict()

        torch, AutoModelForCausalLM, AutoTokenizer = (
            _import_torch_and_transformers()
        )

        sae = _load_sae_full(self.sae_checkpoint)

        device = _resolve_device(torch, self.device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(self.model_name)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        model.to(device)

        layer = int(self.layer)
        captured: list[np.ndarray] = []

        def _capture_hook(module, args):
            captured.append(args[0].detach().cpu().numpy())

        handle = _get_layer_module(model, layer).register_forward_pre_hook(
            _capture_hook
        )
        all_residuals: list[np.ndarray] = []
        all_baseline_logits: list[np.ndarray] = []
        try:
            for prompt in self.prompts:
                captured.clear()
                toks = tokenizer(prompt, return_tensors="pt")
                toks = {k: v.to(device) for k, v in toks.items()}
                with torch.no_grad():
                    out = model(**toks)
                all_residuals.append(captured[0][0].astype(np.float32))
                all_baseline_logits.append(
                    out.logits[0].detach().cpu().numpy().astype(np.float32)
                )
        finally:
            handle.remove()

        residuals = np.concatenate(all_residuals, axis=0)
        baseline_logits = np.concatenate(all_baseline_logits, axis=0)
        n_tokens = int(residuals.shape[0])
        prompt_seq_lens = [int(r.shape[0]) for r in all_residuals]

        # Encode the selected features only (we don't need the full
        # 24576 columns for stat computation — just the panel).
        w_enc = sae["W_enc"]            # (d_model, n_features_total)
        b_enc = sae["b_enc"]            # (n_features_total,)
        b_dec = sae["b_dec"]            # (d_model,)
        w_dec_full = sae["W_dec"]       # (n_features_total, d_model)

        f_per_feature: dict[int, np.ndarray] = {}
        for fid in self.feature_ids:
            pre = (residuals - b_dec) @ w_enc[:, fid] + b_enc[fid]
            f_per_feature[int(fid)] = np.maximum(pre, 0.0).astype(np.float32)

        # Firing-rate sanity warnings.
        for fid in self.feature_ids:
            rate = float((f_per_feature[fid] > 0).mean())
            if rate < self.min_firing_rate:
                warnings.warn(
                    f"BehaviouralValidator: feature {fid} fires on "
                    f"{rate:.4f} of {n_tokens} tokens, below "
                    f"min_firing_rate={self.min_firing_rate}; "
                    f"its Jaccard rows will be near zero. Consider "
                    f"revising feature selection.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Per-feature ablation pass (the cost-cap contract: exactly
        # len(feature_ids) ablation forward-pass-batches).
        kl_per_feature: dict[int, np.ndarray] = {}
        for fid in self.feature_ids:
            kl_per_feature[int(fid)] = _ablation_kl_for_feature(
                model=model,
                tokenizer=tokenizer,
                torch_module=torch,
                layer=layer,
                fid=int(fid),
                w_dec_row=w_dec_full[fid],
                f_activations=f_per_feature[int(fid)],
                baseline_logits=baseline_logits,
                prompts=self.prompts,
                prompt_seq_lens=prompt_seq_lens,
                device=device,
            )

        # Aggregate per pair from cached arrays.
        out_pairs = self._aggregate_pairs(
            candidates,
            f_per_feature=f_per_feature,
            kl_per_feature=kl_per_feature,
        )

        summary = self._summarize(out_pairs)
        confirmed = tuple((p.i, p.j) for p in out_pairs if p.gate_pass)

        return ValidationReport(
            schema_version=SCHEMA_VERSION,
            dictionary_name=self.dictionary.name,
            model_name=self.model_name,
            layer=layer,
            n_prompts=len(list(self.prompts)),
            n_tokens=n_tokens,
            polygram_overlap_threshold=float(self.polygram_overlap_threshold),
            jaccard_threshold=float(self.jaccard_threshold),
            min_firing_rate=float(self.min_firing_rate),
            min_both_fire=int(self.min_both_fire),
            feature_ids=tuple(int(fid) for fid in self.feature_ids),
            pairs=tuple(out_pairs),
            summary=summary,
            confirmed=confirmed,
        )

    def run(self) -> ValidationReport:
        """`validate(predict())` — the convenience wrapper."""
        return self.validate(self.predict())

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _aggregate_pairs(
        self,
        candidates: list[CandidatePair],
        *,
        f_per_feature: dict[int, np.ndarray],
        kl_per_feature: dict[int, np.ndarray],
    ) -> list[CandidatePair]:
        out: list[CandidatePair] = []
        for c in candidates:
            f_i = f_per_feature[int(c.i)]
            f_j = f_per_feature[int(c.j)]
            fires_i = f_i > 0.0
            fires_j = f_j > 0.0
            both = fires_i & fires_j
            either = fires_i | fires_j
            n_both = int(both.sum())
            n_either = int(either.sum())
            n_fires_i = int(fires_i.sum())
            n_fires_j = int(fires_j.sum())

            jaccard = n_both / max(n_either, 1)

            if (
                np.std(f_i) > 1e-12
                and np.std(f_j) > 1e-12
                and f_i.size >= 2
            ):
                pearson_act = float(np.corrcoef(f_i, f_j)[0, 1])
            else:
                pearson_act = float("nan")

            kl_i = kl_per_feature[int(c.i)]
            kl_j = kl_per_feature[int(c.j)]
            kl_i_on_self = (
                float(kl_i[fires_i].mean()) if n_fires_i > 0 else float("nan")
            )
            kl_j_on_self = (
                float(kl_j[fires_j].mean()) if n_fires_j > 0 else float("nan")
            )

            if n_both >= int(self.min_both_fire):
                kl_i_paired_mean = float(kl_i[both].mean())
                kl_j_paired_mean = float(kl_j[both].mean())
                if kl_i_paired_mean > 0.0 and kl_j_paired_mean > 0.0:
                    kl_ratio = kl_i_paired_mean / kl_j_paired_mean
                else:
                    kl_ratio = float("nan")
            else:
                kl_ratio = float("nan")
            kl_log_abs = _safe_log_abs(kl_ratio)

            gate = bool(
                (c.polygram_overlap >= self.polygram_overlap_threshold)
                and (jaccard >= self.jaccard_threshold)
                and (n_both >= int(self.min_both_fire))
            )

            out.append(CandidatePair(
                i=int(c.i),
                j=int(c.j),
                polygram_overlap=float(c.polygram_overlap),
                decoder_overlap=float(c.decoder_overlap),
                jaccard=float(jaccard),
                pearson_activation=pearson_act,
                kl_ablate_i=kl_i_on_self,
                kl_ablate_j=kl_j_on_self,
                kl_ratio_paired=kl_ratio,
                kl_log_ratio_abs=kl_log_abs,
                n_fires_i=n_fires_i,
                n_fires_j=n_fires_j,
                n_both_fire=n_both,
                n_either_fire=n_either,
                gate_pass=gate,
            ))
        return out

    def _summarize(
        self, pairs: list[CandidatePair]
    ) -> ValidationSummary:
        if not pairs:
            empty_buckets = {
                "low_overlap": BucketStats(
                    polygram_range="\u2264 0.4",
                    n_pairs=0,
                    jaccard_mean=float("nan"),
                    jaccard_ci_95=(float("nan"), float("nan")),
                ),
                "mid_overlap": BucketStats(
                    polygram_range="(0.4, 0.7)",
                    n_pairs=0,
                    jaccard_mean=float("nan"),
                    jaccard_ci_95=(float("nan"), float("nan")),
                ),
                "high_overlap": BucketStats(
                    polygram_range="\u2265 0.7",
                    n_pairs=0,
                    jaccard_mean=float("nan"),
                    jaccard_ci_95=(float("nan"), float("nan")),
                ),
            }
            return ValidationSummary(
                spearman_polygram_jaccard=float("nan"),
                spearman_decoder_jaccard=float("nan"),
                spearman_polygram_log_kl_abs=float("nan"),
                pearson_polygram_jaccard=float("nan"),
                pearson_decoder_jaccard=float("nan"),
                buckets=empty_buckets,
                outcome="undefined",
            )

        polygram = np.array([p.polygram_overlap for p in pairs])
        decoder = np.array([p.decoder_overlap for p in pairs])
        jaccard = np.array([p.jaccard for p in pairs])
        log_kl = np.array([p.kl_log_ratio_abs for p in pairs], dtype=float)

        valid_kl = ~np.isnan(log_kl)
        if valid_kl.sum() >= 2:
            sp_kl = _spearman(polygram[valid_kl], log_kl[valid_kl])
        else:
            sp_kl = float("nan")

        sp_pj = _spearman(polygram, jaccard)
        sp_dj = _spearman(decoder, jaccard)
        pe_pj = _pearson(polygram, jaccard)
        pe_dj = _pearson(decoder, jaccard)

        low = jaccard[polygram <= 0.4]
        mid = jaccard[(polygram > 0.4) & (polygram < 0.7)]
        hi = jaccard[polygram >= 0.7]
        buckets = {
            "low_overlap": BucketStats(
                polygram_range="\u2264 0.4",
                n_pairs=int(low.size),
                jaccard_mean=float(low.mean()) if low.size else float("nan"),
                jaccard_ci_95=_bootstrap_ci_mean(low),
            ),
            "mid_overlap": BucketStats(
                polygram_range="(0.4, 0.7)",
                n_pairs=int(mid.size),
                jaccard_mean=float(mid.mean()) if mid.size else float("nan"),
                jaccard_ci_95=_bootstrap_ci_mean(mid),
            ),
            "high_overlap": BucketStats(
                polygram_range="\u2265 0.7",
                n_pairs=int(hi.size),
                jaccard_mean=float(hi.mean()) if hi.size else float("nan"),
                jaccard_ci_95=_bootstrap_ci_mean(hi),
            ),
        }

        if np.isnan(sp_pj):
            outcome = "undefined"
        elif sp_pj >= 0.6:
            outcome = "high_spearman_loop_unblocked"
        elif sp_pj >= 0.3:
            outcome = "medium_spearman_loop_needs_calibration"
        else:
            outcome = "low_spearman_loop_blocked"

        return ValidationSummary(
            spearman_polygram_jaccard=sp_pj,
            spearman_decoder_jaccard=sp_dj,
            spearman_polygram_log_kl_abs=sp_kl,
            pearson_polygram_jaccard=pe_pj,
            pearson_decoder_jaccard=pe_dj,
            buckets=buckets,
            outcome=outcome,
        )


# ============================================================================
# Module-private helpers
# ============================================================================


def _load_sae_full(path: Path) -> dict[str, np.ndarray]:
    """Load `W_enc / b_enc / W_dec / b_dec` from a `.safetensors` file
    via the safetensors loader. Raises `ValueError` listing missing
    tensors on incomplete files.
    """
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "BehaviouralValidator: safetensors is required to load "
            "the SAE checkpoint; install via `pip install polygram[sae]`."
        ) from exc

    out: dict[str, np.ndarray] = {}
    with safe_open(str(path), framework="numpy") as f:
        present = set(f.keys())
        required = ("W_enc", "b_enc", "W_dec", "b_dec")
        missing = [k for k in required if k not in present]
        if missing:
            raise ValueError(
                f"BehaviouralValidator: SAE checkpoint at {path} is "
                f"missing tensor(s) {missing}; required keys are "
                f"{list(required)}, present: {sorted(present)}"
            )
        for k in required:
            out[k] = np.asarray(f.get_tensor(k), dtype=np.float32)
    return out


def _ablation_kl_for_feature(
    *,
    model,
    tokenizer,
    torch_module,
    layer: int,
    fid: int,
    w_dec_row: np.ndarray,
    f_activations: np.ndarray,
    baseline_logits: np.ndarray,
    prompts: Sequence[str],
    prompt_seq_lens: Sequence[int],
    device: str = "cpu",
) -> np.ndarray:
    """Run a single ablation forward-pass-batch for one feature.

    For every prompt, registers a forward-pre-hook that subtracts
    `f * w_dec_row` at every token where this feature fires; runs the
    forward; collects per-token KL between baseline and ablated
    next-token logits.

    Returns a per-token KL array of length `sum(prompt_seq_lens)`.
    """
    n_tokens_total = int(np.asarray(prompt_seq_lens).sum())
    kl_per_token = np.zeros(n_tokens_total, dtype=np.float32)
    token_offset = 0
    for prompt, seq_len in zip(prompts, prompt_seq_lens):
        f_slice = f_activations[token_offset : token_offset + seq_len]
        base_slice = baseline_logits[token_offset : token_offset + seq_len]
        fires_idx = np.where(f_slice > 0)[0]
        if fires_idx.size == 0:
            token_offset += seq_len
            continue

        toks = tokenizer(prompt, return_tensors="pt")
        toks = {k: v.to(device) for k, v in toks.items()}
        # Re-tokenize and confirm length agreement; we already did this
        # to build f_activations via the capture pass.
        # (Tokenizer is deterministic; lengths align by construction.)

        def _ablate_hook(module, args, _fires=fires_idx, _f=f_slice,
                         _w=w_dec_row):
            h = args[0]
            h_np = h.detach().cpu().numpy()[0].copy()
            for t in _fires:
                h_np[t] = h_np[t] - float(_f[t]) * _w
            new_h = torch_module.from_numpy(
                h_np[None, ...]
            ).to(h.dtype).to(h.device)
            return (new_h,) + args[1:]

        handle = _get_layer_module(model, layer).register_forward_pre_hook(
            _ablate_hook
        )
        try:
            with torch_module.no_grad():
                out = model(**toks)
        finally:
            handle.remove()
        ablated_logits = (
            out.logits[0].detach().cpu().numpy().astype(np.float32)
        )
        for t in fires_idx:
            kl_per_token[token_offset + int(t)] = _kl_softmax_row(
                base_slice[t], ablated_logits[t]
            )
        token_offset += seq_len
    return kl_per_token
