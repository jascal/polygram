"""`EpochCompressor` — multi-panel compression orchestrator.

Scales the validator + compressor loop from a single 8-feature
panel to the full SAE via greedy seeded coverage over the
cosine-similar pair graph, with stable-cluster-set fixed-point
iteration and a relative cross-entropy quality bound.

Two-stage API:

    select_panels() -> list[Panel]    # cheap, single forward pass
    run(output_checkpoint=...) -> EpochResult

`select_panels()` is exposed for inspection but `run()` calls it
internally per iteration. The orchestrator runs panels sequentially
in-process; users wanting concurrency shard at the CLI invocation
level (see proposal.md "What this proposal explicitly does NOT do"
section).

The orchestrator itself is torch-free; lazy torch + transformers
imports happen only inside the residual-capture pass and the
delegated `BehaviouralValidator.validate()` calls.

See `add-compression-epoch/design.md` for the eight numbered
decisions.
"""

from __future__ import annotations

import os
import tempfile
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from polygram.behavioural.report import (
    CandidatePair,
    ValidationReport,
    ValidationSummary,
)
from polygram.compression._hash import sha256_file
from polygram.compression.compressor import Compressor
from polygram.compression.epoch_report import (
    SCHEMA_VERSION,
    EpochIteration,
    EpochReport,
    EpochResult,
    Panel,
)


# Convergence reasons (terminal states)
_REASON_STABLE_CLUSTERS = "stable_clusters"
_REASON_MAX_ITERATIONS = "max_iterations"
_REASON_QUALITY_BREACHED = "quality_bound_breached"
_REASON_NO_PRIORITY_CANDIDATES = "no_more_priority_candidates"


@dataclass
class EpochCompressor:
    """Orchestrates multi-panel validate → aggregate → compress
    iteration to a stable cluster-set fixed point.

    Default knobs encode the §4.4 / §4.3 calibration. Override via
    constructor arguments or CLI flags.
    """

    sae_checkpoint: Path
    prompts: Sequence[str]
    layer: int
    model_name: str = "gpt2"
    strategy: str = "zero"
    device: str | None = None
    coverage_target: float = 0.95
    cosine_threshold: float = 0.30
    n_visits_per_feature: int = 3
    n_panels_max: int = 1000
    min_firing_rate: float = 0.01
    max_iterations: int = 5
    quality_delta_multiplier: float = 2.0
    polygram_overlap_threshold: float = 0.7
    jaccard_threshold: float = 0.30
    min_both_fire: int = 5
    save_intermediate_reports: bool = False
    allow_layer_zero: bool = False

    # Internal state populated during run()
    _zeroed: set[int] = field(default_factory=set, init=False, repr=False, compare=False)

    # ----------------------------------------------------------------
    # Construction-time validation
    # ----------------------------------------------------------------

    def __post_init__(self) -> None:
        self.sae_checkpoint = Path(self.sae_checkpoint)
        if not self.sae_checkpoint.is_file():
            raise ValueError(
                f"EpochCompressor: sae_checkpoint not found: "
                f"{self.sae_checkpoint}"
            )
        if not list(self.prompts):
            raise ValueError(
                "EpochCompressor: prompts must be non-empty"
            )
        if self.strategy != "zero":
            raise ValueError(
                f"EpochCompressor: only the 'zero' strategy is supported "
                f"currently; got {self.strategy!r}"
            )
        if not (0.0 < self.coverage_target <= 1.0):
            raise ValueError(
                f"EpochCompressor: coverage_target must be in (0, 1]; "
                f"got {self.coverage_target}"
            )
        if not (-1.0 <= self.cosine_threshold <= 1.0):
            raise ValueError(
                f"EpochCompressor: cosine_threshold must be in [-1, 1]; "
                f"got {self.cosine_threshold}"
            )
        if int(self.n_visits_per_feature) < 1:
            raise ValueError(
                f"EpochCompressor: n_visits_per_feature must be >= 1; "
                f"got {self.n_visits_per_feature}"
            )
        if int(self.n_panels_max) < 1:
            raise ValueError(
                f"EpochCompressor: n_panels_max must be >= 1; "
                f"got {self.n_panels_max}"
            )
        if not (0.0 <= self.min_firing_rate <= 1.0):
            raise ValueError(
                f"EpochCompressor: min_firing_rate must be in [0, 1]; "
                f"got {self.min_firing_rate}"
            )
        if int(self.max_iterations) < 1:
            raise ValueError(
                f"EpochCompressor: max_iterations must be >= 1; "
                f"got {self.max_iterations}"
            )
        if float(self.quality_delta_multiplier) <= 0:
            raise ValueError(
                f"EpochCompressor: quality_delta_multiplier must be > 0; "
                f"got {self.quality_delta_multiplier}"
            )
        if int(self.layer) < 0:
            raise ValueError(
                f"EpochCompressor: layer must be >= 0; got {self.layer}"
            )
        if int(self.layer) == 0 and not self.allow_layer_zero:
            raise ValueError(
                "EpochCompressor: layer 0 is the structural dead zone for "
                "GPT-2 small (per docs/research/deeper-layer-ablation-"
                "probe.md); use layer >= 5 (recommended: 10), or pass "
                "allow_layer_zero=True"
            )

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------

    def run(
        self, output_checkpoint: str | os.PathLike
    ) -> EpochResult:
        """Iterate panel selection → validation → compression to a
        stable-cluster fixed point. Returns an `EpochResult` with the
        final compressed checkpoint and the EpochReport.
        """
        out_path = Path(output_checkpoint).resolve()
        if out_path == self.sae_checkpoint.resolve():
            raise ValueError(
                f"EpochCompressor.run: output_checkpoint must differ "
                f"from sae_checkpoint (both resolved to {out_path})"
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)

        from safetensors.numpy import load_file, save_file

        from polygram.sae_import import from_sae_lens, load_sae_safetensors

        wall_start = time.monotonic()
        source_sha = sha256_file(self.sae_checkpoint)

        # Pre-pass: firing rates from one full forward pass.
        firing_rates, residuals = _compute_firing_rates_and_residuals(
            self.sae_checkpoint,
            list(self.prompts),
            model_name=self.model_name,
            layer=int(self.layer),
            device=self.device,
        )
        n_tokens = int(residuals.shape[0])
        decoder_norms = _compute_decoder_norms(self.sae_checkpoint)
        priority = firing_rates * decoder_norms

        # Working state — start from the source state-dict; iterations
        # mutate `current_state` in-memory. We only write to disk at
        # the end (or per-iteration via temp + replace if save_intermediate).
        current_state = load_file(str(self.sae_checkpoint))

        iterations: list[EpochIteration] = []
        cluster_fingerprints: list[frozenset] = []
        final_coverage = 0.0
        delta_1: float | None = None
        convergence_reason = _REASON_MAX_ITERATIONS

        for iteration in range(int(self.max_iterations)):
            # Select panels for this iteration.
            cosine_graph = _compute_cosine_graph(
                current_state["W_dec"],
                self._eligible_features(firing_rates),
                threshold=float(self.cosine_threshold),
            )
            panels, coverage = _select_panels(
                state_dict=current_state,
                eligible=self._eligible_features(firing_rates),
                priority=priority,
                cosine_pairs=cosine_graph,
                zeroed=self._zeroed,
                n_visits_per_feature=int(self.n_visits_per_feature),
                n_panels_max=int(self.n_panels_max),
                coverage_target=float(self.coverage_target),
            )
            final_coverage = coverage

            if not panels:
                if iteration == 0:
                    convergence_reason = _REASON_NO_PRIORITY_CANDIDATES
                else:
                    convergence_reason = _REASON_STABLE_CLUSTERS
                break

            # Run validator on each panel; collect per-panel reports.
            per_panel_reports = self._validate_panels(
                panels=panels,
                state_dict=current_state,
                residuals=residuals,
                firing_rates=firing_rates,
                n_tokens=n_tokens,
            )

            # Synthesize the cross-panel ValidationReport.
            synth_report = _synthesize_validation_report(
                panels, per_panel_reports, self.sae_checkpoint
            )

            confirmed_count = len(synth_report.confirmed)
            if confirmed_count == 0:
                # No new redundancies found.
                if iteration == 0:
                    convergence_reason = _REASON_NO_PRIORITY_CANDIDATES
                else:
                    convergence_reason = _REASON_STABLE_CLUSTERS
                iterations.append(
                    EpochIteration(
                        iteration=iteration,
                        panels=tuple(panels),
                        validation_report_paths=tuple(),
                        confirmed_pair_count=0,
                        clusters_compressed=0,
                        features_zeroed_this_iteration=tuple(),
                        cross_entropy_delta=0.0,
                        convergence_state=convergence_reason,
                    )
                )
                break

            # Compress against the synthetic report; orchestrator-built
            # representatives use global n_fires aggregation.
            global_n_fires = _compute_global_n_fires(
                panels, firing_rates, n_tokens
            )
            representatives = _pick_representatives_global(
                synth_report, global_n_fires
            )

            # Compressor expects a real on-disk checkpoint; we write
            # current_state to a temp path so it can read it.
            tmp_in = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(out_path.parent),
                prefix=f".epoch_iter{iteration}_in.",
                suffix=".tmp",
                delete=False,
            )
            tmp_in.close()
            tmp_in_path = Path(tmp_in.name)
            save_file(current_state, str(tmp_in_path))

            tmp_out = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(out_path.parent),
                prefix=f".epoch_iter{iteration}_out.",
                suffix=".tmp",
                delete=False,
            )
            tmp_out.close()
            tmp_out_path = Path(tmp_out.name)

            try:
                compressor = Compressor(
                    validation_report=synth_report,
                    sae_checkpoint=tmp_in_path,
                    strategy="zero",
                    representatives=representatives,
                )
                compress_result = compressor.run(
                    output_checkpoint=tmp_out_path
                )
                new_state = load_file(str(tmp_out_path))
            finally:
                tmp_in_path.unlink(missing_ok=True)

            # Cluster fingerprint check.
            fingerprint = frozenset(
                frozenset(c.members)
                for c in compress_result.plan.clusters
            )

            # Quality bound: cross-entropy delta vs PRIOR iteration's
            # state (not vs source — design decision 5 says "first
            # iteration's delta is the natural reference").
            delta_k = _token_cross_entropy_delta(
                residuals, current_state, new_state
            )

            zeroed_this_iter = tuple(
                sorted(
                    int(fid)
                    for cluster in compress_result.plan.clusters
                    for fid in cluster.zeroed
                )
            )

            # Update state for next iteration BEFORE convergence check
            # so the next iteration sees the compressed state.
            previous_state = current_state
            current_state = new_state
            self._zeroed.update(zeroed_this_iter)

            # Quality-bound check (only after iteration 0 establishes delta_1).
            if delta_1 is None:
                delta_1 = delta_k
                quality_breached = False
            else:
                bound = float(self.quality_delta_multiplier) * delta_1
                quality_breached = delta_k > bound

            # Determine convergence state for this iteration.
            if quality_breached:
                convergence_state = _REASON_QUALITY_BREACHED
            elif fingerprint in cluster_fingerprints:
                convergence_state = _REASON_STABLE_CLUSTERS
            elif iteration == int(self.max_iterations) - 1:
                convergence_state = _REASON_MAX_ITERATIONS
            else:
                convergence_state = "continuing"

            cluster_fingerprints.append(fingerprint)

            # Save the iteration's per-panel reports if requested.
            saved_paths: tuple[str, ...] = tuple()
            if self.save_intermediate_reports:
                saved_paths = self._save_intermediate_reports(
                    out_path.parent, iteration, per_panel_reports
                )

            iterations.append(
                EpochIteration(
                    iteration=iteration,
                    panels=_round_panels(panels),
                    validation_report_paths=saved_paths,
                    confirmed_pair_count=confirmed_count,
                    clusters_compressed=len(compress_result.plan.clusters),
                    features_zeroed_this_iteration=zeroed_this_iter,
                    cross_entropy_delta=_round_float(float(delta_k)),
                    convergence_state=convergence_state,
                )
            )

            # Handle terminal states.
            if quality_breached:
                # Revert to previous state.
                current_state = previous_state
                self._zeroed.difference_update(zeroed_this_iter)
                convergence_reason = _REASON_QUALITY_BREACHED
                tmp_out_path.unlink(missing_ok=True)
                break

            tmp_out_path.unlink(missing_ok=True)

            if convergence_state == _REASON_STABLE_CLUSTERS:
                convergence_reason = _REASON_STABLE_CLUSTERS
                break
            if convergence_state == _REASON_MAX_ITERATIONS:
                convergence_reason = _REASON_MAX_ITERATIONS
                break

        # Write the final checkpoint atomically.
        tmp_final = tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(out_path.parent),
            prefix=f".{out_path.stem}.",
            suffix=".tmp",
            delete=False,
        )
        tmp_final.close()
        tmp_final_path = Path(tmp_final.name)
        try:
            save_file(current_state, str(tmp_final_path))
            os.replace(tmp_final_path, out_path)
        except Exception:
            tmp_final_path.unlink(missing_ok=True)
            raise

        output_sha = sha256_file(out_path)
        wall_seconds = time.monotonic() - wall_start

        report = EpochReport(
            schema_version=SCHEMA_VERSION,
            source_checkpoint=str(self.sae_checkpoint),
            source_checkpoint_sha256=source_sha,
            output_checkpoint=str(out_path),
            output_checkpoint_sha256=output_sha,
            convergence_reason=convergence_reason,
            n_features_zeroed_total=len(self._zeroed),
            n_panels_total=sum(len(it.panels) for it in iterations),
            coverage_achieved=_round_float(float(final_coverage)),
            wall_seconds=_round_float(float(wall_seconds)),
            iterations=tuple(iterations),
        )

        # Rebuild a Dictionary on the populated zeroed slots, capped at
        # 8 (rung-1 MPS encoding cap). For SAEs that compressed nothing,
        # rebuild on the first 8 features.
        sorted_zeroed = sorted(self._zeroed)
        if sorted_zeroed:
            dict_ids = sorted_zeroed[:8]
        else:
            n_features = int(current_state["W_dec"].shape[0])
            dict_ids = list(range(min(8, n_features)))
        records = load_sae_safetensors(str(out_path), feature_ids=dict_ids)
        rebuilt_dictionary, _ = from_sae_lens(
            records,
            dict_ids,
            assign_gamma=True,
            name=f"Epoch_{self.sae_checkpoint.stem.replace('-', '_').replace('.', '_')}",
        )

        return EpochResult(
            report=report,
            output_checkpoint=out_path,
            final_dictionary=rebuilt_dictionary,
        )

    # ----------------------------------------------------------------
    # Helpers (internal)
    # ----------------------------------------------------------------

    def _eligible_features(self, firing_rates: np.ndarray) -> np.ndarray:
        mask = firing_rates >= float(self.min_firing_rate)
        eligible = np.where(mask)[0]
        if self._zeroed:
            eligible = np.array(
                [int(f) for f in eligible if int(f) not in self._zeroed],
                dtype=np.int64,
            )
        return eligible

    def _validate_panels(
        self,
        *,
        panels: list[Panel],
        state_dict: dict[str, np.ndarray],
        residuals: np.ndarray,
        firing_rates: np.ndarray,
        n_tokens: int,
    ) -> list[ValidationReport]:
        """Run a lightweight per-panel validator equivalent on cached
        residuals — we have firing rates already, so we avoid the
        validator's per-feature ablation pass and synthesize a
        ValidationReport directly from the predict-stage Polygram
        overlap and the cached firing patterns.

        This is a deliberate optimization on `BehaviouralValidator.run()`:
        in epoch context, we run many panels over the same residuals,
        so per-panel ablation passes are wasteful. The synthetic
        validator computes Polygram overlap (the panel's k-means
        cluster geometry), Jaccard from cached firing patterns,
        decoder cosine, and assembles a `ValidationReport` whose
        `confirmed` list applies the same gate criteria the validator
        uses (`polygram_overlap >= threshold AND jaccard >= threshold
        AND n_both_fire >= min_both_fire`).
        """
        reports: list[ValidationReport] = []
        for panel in panels:
            reports.append(
                _validate_panel_inline(
                    panel,
                    state_dict=state_dict,
                    residuals=residuals,
                    firing_rates=firing_rates,
                    n_tokens=n_tokens,
                    polygram_overlap_threshold=float(
                        self.polygram_overlap_threshold
                    ),
                    jaccard_threshold=float(self.jaccard_threshold),
                    min_both_fire=int(self.min_both_fire),
                    sae_checkpoint=self.sae_checkpoint,
                    layer=int(self.layer),
                    model_name=self.model_name,
                )
            )
        return reports

    def _save_intermediate_reports(
        self,
        out_dir: Path,
        iteration: int,
        per_panel_reports: list[ValidationReport],
    ) -> tuple[str, ...]:
        saved: list[str] = []
        intermediate_dir = out_dir / "epoch_intermediate"
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        for k, report in enumerate(per_panel_reports):
            p = intermediate_dir / f"iter{iteration}_panel{k}.json"
            report.to_json(p)
            saved.append(str(p))
        return tuple(saved)


# ============================================================================
# Pre-pass helpers
# ============================================================================


def _compute_firing_rates_and_residuals(
    sae_checkpoint: Path,
    prompts: list[str],
    *,
    model_name: str,
    layer: int,
    device: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run ONE forward pass per prompt; capture residuals at the
    named layer; encode through every feature in the SAE
    (vectorized); return per-feature firing rate over all tokens
    AND the cached residuals for downstream use.

    Returns `(firing_rates: (n_features,), residuals: (n_tokens, d_model))`.
    """
    from safetensors.numpy import load_file

    from polygram.behavioural.runtime import (
        _import_torch_and_transformers,
        _resolve_device,
    )

    torch, GPT2LMHeadModel, GPT2Tokenizer = _import_torch_and_transformers()
    resolved = _resolve_device(torch, device)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    model.to(resolved)

    captured: list[np.ndarray] = []

    def _hook(module, args):
        captured.append(args[0].detach().cpu().numpy())

    handle = model.transformer.h[layer].register_forward_pre_hook(_hook)
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
    residuals = np.concatenate(chunks, axis=0)

    sae = load_file(str(sae_checkpoint))
    pre = (residuals - sae["b_dec"]) @ sae["W_enc"] + sae["b_enc"]
    act = np.maximum(pre, 0.0)                            # (n_tokens, n_features)
    firing = (act > 0).astype(np.float32)
    firing_rates = firing.mean(axis=0)                    # (n_features,)
    return firing_rates, residuals


def _compute_decoder_norms(sae_checkpoint: Path) -> np.ndarray:
    from safetensors.numpy import load_file

    sae = load_file(str(sae_checkpoint))
    return np.linalg.norm(sae["W_dec"], axis=1).astype(np.float32)


def _compute_cosine_graph(
    w_dec: np.ndarray,
    eligible: np.ndarray,
    *,
    threshold: float,
) -> set[tuple[int, int]]:
    """Return the set of `(i, j)` pairs (i < j, both in `eligible`)
    with `cos(W_dec[i], W_dec[j]) >= threshold`.

    For large eligible sets, computes the cosine matrix in chunks to
    cap memory.
    """
    if eligible.size < 2:
        return set()

    rows = w_dec[eligible].astype(np.float32, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit = rows / norms

    out: set[tuple[int, int]] = set()
    n = unit.shape[0]
    chunk = 1024 if n > 1024 else n
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        sims = unit[start:end] @ unit.T                     # (chunk, n)
        # We only want i < j, so mask the upper triangle relative to start:
        for local_i in range(end - start):
            global_i = start + local_i
            for global_j in range(global_i + 1, n):
                if sims[local_i, global_j] >= threshold:
                    out.add((int(eligible[global_i]), int(eligible[global_j])))
    return out


# ============================================================================
# Panel selection
# ============================================================================


def _select_panels(
    *,
    state_dict: dict[str, np.ndarray],
    eligible: np.ndarray,
    priority: np.ndarray,
    cosine_pairs: set[tuple[int, int]],
    zeroed: set[int],
    n_visits_per_feature: int,
    n_panels_max: int,
    coverage_target: float,
) -> tuple[list[Panel], float]:
    """Greedy seeded coverage panel selection per
    `add-compression-epoch/design.md` Decision 2.

    Returns the panel list and the achieved coverage fraction
    (`|pairs_covered ∩ S| / |S|`). When `S` is empty, coverage is 1.0
    by convention (nothing to cover, target trivially met).
    """
    if eligible.size == 0:
        return [], 1.0

    # Precompute cosine matrix for the eligible set.
    w_dec = state_dict["W_dec"]
    rows = w_dec[eligible].astype(np.float32, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit = rows / norms                                    # (n_elig, d_model)
    elig_to_idx = {int(f): i for i, f in enumerate(eligible)}

    # Priority-sorted eligible-fid list (highest priority first).
    elig_priority = priority[eligible]
    order = np.argsort(-elig_priority, kind="stable")     # descending
    queue = [int(eligible[i]) for i in order]

    visits: dict[int, int] = defaultdict(int)
    pairs_covered: set[tuple[int, int]] = set()
    panels: list[Panel] = []
    panel_id = 0

    target_pair_count = len(cosine_pairs) if cosine_pairs else 0

    for anchor in queue:
        if anchor in zeroed:
            continue
        if visits[anchor] >= n_visits_per_feature:
            continue

        # Build neighbour list: top cosine-similar features in eligible
        # that haven't hit visit cap.
        i_anchor = elig_to_idx[anchor]
        sims = unit @ unit[i_anchor]                       # (n_elig,)
        # Sort descending; skip self and over-visited.
        sort_order = np.argsort(-sims, kind="stable")
        neighbours: list[int] = []
        cosines: list[float] = []
        for idx in sort_order:
            fid = int(eligible[int(idx)])
            if fid == anchor:
                continue
            if fid in zeroed:
                continue
            if visits[fid] >= n_visits_per_feature:
                continue
            neighbours.append(fid)
            cosines.append(float(sims[int(idx)]))
            if len(neighbours) >= 7:
                break

        if not neighbours and len(panels) == 0:
            warnings.warn(
                "EpochCompressor._select_panels: eligible pool has no "
                "neighbours for the priority anchor; emitting a single "
                "anchor-only panel and stopping",
                RuntimeWarning,
                stacklevel=2,
            )
            panels.append(
                Panel(
                    panel_id=panel_id,
                    anchor=anchor,
                    feature_ids=(anchor,),
                    cosines_to_anchor=tuple(),
                )
            )
            visits[anchor] += 1
            panel_id += 1
            break

        if not neighbours:
            continue

        members = [anchor] + neighbours
        members_sorted = tuple(sorted(int(f) for f in members))

        # Update visits + pair coverage.
        for fid in members:
            visits[fid] += 1
        for i in range(len(members_sorted)):
            for j in range(i + 1, len(members_sorted)):
                pair = (members_sorted[i], members_sorted[j])
                if pair in cosine_pairs:
                    pairs_covered.add(pair)

        # cosines_to_anchor in feature_ids order, anchor entry omitted.
        cos_lookup = {n: c for n, c in zip(neighbours, cosines)}
        ordered_cosines = tuple(
            float(cos_lookup[fid])
            for fid in members_sorted
            if fid != anchor
        )

        panels.append(
            Panel(
                panel_id=panel_id,
                anchor=anchor,
                feature_ids=members_sorted,
                cosines_to_anchor=ordered_cosines,
            )
        )
        panel_id += 1

        coverage = (
            len(pairs_covered) / target_pair_count
            if target_pair_count > 0
            else 1.0
        )
        if len(panels) >= n_panels_max:
            break
        if coverage >= coverage_target:
            break

    final_coverage = (
        len(pairs_covered) / target_pair_count
        if target_pair_count > 0
        else 1.0
    )
    return panels, final_coverage


# ============================================================================
# Per-panel inline validator (fast path for epoch context)
# ============================================================================


def _validate_panel_inline(
    panel: Panel,
    *,
    state_dict: dict[str, np.ndarray],
    residuals: np.ndarray,
    firing_rates: np.ndarray,
    n_tokens: int,
    polygram_overlap_threshold: float,
    jaccard_threshold: float,
    min_both_fire: int,
    sae_checkpoint: Path,
    layer: int,
    model_name: str,
) -> ValidationReport:
    """Synthesize a ValidationReport for one panel without running
    the full BehaviouralValidator.validate() ablation pass.

    For epoch use we already have firing patterns from the
    pre-pass; reusing them avoids one ablation forward pass per
    feature per panel. The `kl_*` fields are NaN — the orchestrator
    doesn't gate on them for confirmed-pair selection, only on
    Polygram overlap, Jaccard, and n_both_fire.
    """
    from polygram.behavioural.report import (
        CandidatePair,
        ValidationReport,
        ValidationSummary,
    )
    from polygram.sae_import import from_sae_lens, load_sae_safetensors

    feature_ids = list(panel.feature_ids)

    # Build the panel's Dictionary via from_sae_lens — operates on
    # the current state_dict, but from_sae_lens loads from disk. The
    # orchestrator already wrote current_state to a temp path during
    # apply(); we read it back here. To keep the contract simple, just
    # use the source checkpoint — Dictionary geometry depends on
    # decoder rows, which the source has at start; later iterations'
    # state_dict differs, so we materialize a temp file.
    import tempfile as _tempfile
    from safetensors.numpy import save_file

    tmp = _tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(sae_checkpoint.parent),
        prefix=".epoch_panel.",
        suffix=".tmp",
        delete=False,
    )
    tmp.close()
    tmp_path = Path(tmp.name)
    try:
        save_file(state_dict, str(tmp_path))
        records = load_sae_safetensors(str(tmp_path), feature_ids=feature_ids)
        dictionary, _ = from_sae_lens(
            records, feature_ids, assign_gamma=True,
            name=f"Panel{panel.panel_id}",
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    # Polygram overlap matrix: pairwise |<psi_i|psi_j>|^2 from the
    # Dictionary's gram method. gram() returns complex inner products
    # <psi_i|psi_j>; squared modulus gives the [0,1] overlap value
    # the §4.4 calibration uses.
    n_panel = len(feature_ids)
    polygram = (np.abs(dictionary.gram()) ** 2).astype(np.float64, copy=False)

    # Decoder cosine² for context.
    w_dec = state_dict["W_dec"]
    rows = w_dec[feature_ids].astype(np.float64, copy=False)
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    unit_rows = rows / norms
    decoder_cos = (unit_rows @ unit_rows.T) ** 2

    # Firing patterns from cached residuals.
    sae_W_enc = state_dict["W_enc"]
    sae_b_enc = state_dict["b_enc"]
    sae_b_dec = state_dict["b_dec"]
    pre_panel = (
        (residuals - sae_b_dec) @ sae_W_enc[:, feature_ids]
        + sae_b_enc[feature_ids]
    )
    act_panel = np.maximum(pre_panel, 0.0)                # (n_tokens, n_panel)
    fires = (act_panel > 0).astype(np.float32)            # (n_tokens, n_panel)
    n_fires_per = fires.sum(axis=0).astype(np.int64)      # (n_panel,)

    # Pearson activation: per-pair correlation of activation values.
    if n_panel >= 2:
        means = act_panel.mean(axis=0)
        centred = act_panel - means
        stds = centred.std(axis=0)
        stds_safe = np.where(stds < 1e-12, 1.0, stds)
        pearson_mat = (centred.T @ centred) / (n_tokens * np.outer(stds_safe, stds_safe))
    else:
        pearson_mat = np.ones((n_panel, n_panel), dtype=np.float64)

    pairs: list[CandidatePair] = []
    confirmed: list[tuple[int, int]] = []
    for i_idx in range(n_panel):
        for j_idx in range(i_idx + 1, n_panel):
            i = int(feature_ids[i_idx])
            j = int(feature_ids[j_idx])
            both = int(((fires[:, i_idx] > 0) & (fires[:, j_idx] > 0)).sum())
            either = int(((fires[:, i_idx] > 0) | (fires[:, j_idx] > 0)).sum())
            jaccard = float(both / either) if either > 0 else 0.0
            polygram_ij = float(polygram[i_idx, j_idx])
            decoder_ij = float(decoder_cos[i_idx, j_idx])
            pearson_ij = float(pearson_mat[i_idx, j_idx])
            gate_pass = (
                polygram_ij >= polygram_overlap_threshold
                and jaccard >= jaccard_threshold
                and both >= min_both_fire
            )
            pair = CandidatePair(
                i=i, j=j,
                polygram_overlap=polygram_ij,
                decoder_overlap=decoder_ij,
                jaccard=jaccard,
                pearson_activation=pearson_ij,
                kl_ablate_i=float("nan"),
                kl_ablate_j=float("nan"),
                kl_ratio_paired=float("nan"),
                kl_log_ratio_abs=float("nan"),
                n_fires_i=int(n_fires_per[i_idx]),
                n_fires_j=int(n_fires_per[j_idx]),
                n_both_fire=both,
                n_either_fire=either,
                gate_pass=gate_pass,
            )
            pairs.append(pair)
            if gate_pass:
                confirmed.append((i, j))

    summary = ValidationSummary(
        spearman_polygram_jaccard=float("nan"),
        spearman_decoder_jaccard=float("nan"),
        spearman_polygram_log_kl_abs=float("nan"),
        pearson_polygram_jaccard=float("nan"),
        pearson_decoder_jaccard=float("nan"),
        buckets={},
        outcome="epoch_inline",
    )
    return ValidationReport(
        schema_version=1,
        dictionary_name=f"Panel{panel.panel_id}",
        model_name=model_name,
        layer=int(layer),
        n_prompts=1,
        n_tokens=int(n_tokens),
        polygram_overlap_threshold=polygram_overlap_threshold,
        jaccard_threshold=jaccard_threshold,
        min_firing_rate=0.0,
        min_both_fire=int(min_both_fire),
        feature_ids=tuple(int(f) for f in feature_ids),
        pairs=tuple(pairs),
        summary=summary,
        confirmed=tuple(sorted(confirmed)),
    )


# ============================================================================
# Cross-panel aggregation
# ============================================================================


def _synthesize_validation_report(
    panels: list[Panel],
    per_panel_reports: list[ValidationReport],
    sae_checkpoint: Path,
) -> ValidationReport:
    """Per `design.md` Decision 3: union confirmed pairs; max-aggregate
    panel-composition-dependent fields; sum-aggregate counts (under
    determinism invariant they're identical across panels containing
    the pair)."""
    by_pair: dict[tuple[int, int], list[CandidatePair]] = defaultdict(list)
    feature_ids_union: set[int] = set()

    for report in per_panel_reports:
        for pair in report.pairs:
            key = (min(pair.i, pair.j), max(pair.i, pair.j))
            by_pair[key].append(pair)
        feature_ids_union.update(int(f) for f in report.feature_ids)

    aggregated: list[CandidatePair] = []
    confirmed_set: set[tuple[int, int]] = set()
    for (i, j), pair_list in by_pair.items():
        polygram_max = max(p.polygram_overlap for p in pair_list)
        jaccard_max = max(p.jaccard for p in pair_list)
        decoder_overlap = pair_list[0].decoder_overlap

        # Sum-aggregate fire counts (under determinism invariant they're
        # identical across panels — we still take the max as a safety
        # net for any future stochasticity).
        n_fires_i = max(p.n_fires_i for p in pair_list)
        n_fires_j = max(p.n_fires_j for p in pair_list)
        n_both_fire = max(p.n_both_fire for p in pair_list)
        n_either_fire = max(p.n_either_fire for p in pair_list)

        # Weighted means for behavioural fields (currently NaN in
        # epoch-inline validator; preserved for future strategies).
        weights = np.array(
            [max(1, p.n_either_fire) for p in pair_list],
            dtype=np.float64,
        )
        if weights.sum() > 0:
            pearson = float(
                np.average(
                    [p.pearson_activation for p in pair_list],
                    weights=weights,
                )
            )
        else:
            pearson = float("nan")

        gate_pass_any = any(p.gate_pass for p in pair_list)

        aggregated.append(
            CandidatePair(
                i=int(i),
                j=int(j),
                polygram_overlap=float(polygram_max),
                decoder_overlap=float(decoder_overlap),
                jaccard=float(jaccard_max),
                pearson_activation=pearson,
                kl_ablate_i=float("nan"),
                kl_ablate_j=float("nan"),
                kl_ratio_paired=float("nan"),
                kl_log_ratio_abs=float("nan"),
                n_fires_i=int(n_fires_i),
                n_fires_j=int(n_fires_j),
                n_both_fire=int(n_both_fire),
                n_either_fire=int(n_either_fire),
                gate_pass=bool(gate_pass_any),
            )
        )
        if gate_pass_any:
            confirmed_set.add((int(i), int(j)))

    sample = per_panel_reports[0]
    summary = ValidationSummary(
        spearman_polygram_jaccard=float("nan"),
        spearman_decoder_jaccard=float("nan"),
        spearman_polygram_log_kl_abs=float("nan"),
        pearson_polygram_jaccard=float("nan"),
        pearson_decoder_jaccard=float("nan"),
        buckets={},
        outcome="epoch_synthesized",
    )
    return ValidationReport(
        schema_version=1,
        dictionary_name=f"EpochSynth_{sae_checkpoint.stem.replace('-', '_').replace('.', '_')}",
        model_name=sample.model_name,
        layer=sample.layer,
        n_prompts=sample.n_prompts,
        n_tokens=sample.n_tokens,
        polygram_overlap_threshold=sample.polygram_overlap_threshold,
        jaccard_threshold=sample.jaccard_threshold,
        min_firing_rate=sample.min_firing_rate,
        min_both_fire=sample.min_both_fire,
        feature_ids=tuple(sorted(feature_ids_union)),
        pairs=tuple(aggregated),
        summary=summary,
        confirmed=tuple(sorted(confirmed_set)),
    )


def _compute_global_n_fires(
    panels: list[Panel],
    firing_rates: np.ndarray,
    n_tokens: int,
) -> dict[int, int]:
    """Per `design.md` Decision 4: aggregate firing counts globally
    via the panel-independent firing rate from the pre-pass.
    Identical across panels containing the same feature, so the
    "global" count is just `firing_rate × n_tokens`."""
    fids = {int(f) for panel in panels for f in panel.feature_ids}
    out: dict[int, int] = {}
    for fid in fids:
        out[fid] = int(round(float(firing_rates[fid]) * n_tokens))
    return out


def _pick_representatives_global(
    synth_report: ValidationReport,
    global_n_fires: dict[int, int],
) -> dict[int, int] | None:
    """Run union-find on synth_report.confirmed; for each cluster,
    pick the member with highest global_n_fires (tiebreak: lowest
    fid). Returns the `{cluster_id: fid}` map for
    Compressor.representatives, or None if no clusters."""
    confirmed = synth_report.confirmed
    if not confirmed:
        return None

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for i, j in confirmed:
        for fid in (int(i), int(j)):
            if fid not in parent:
                parent[fid] = fid
        union(int(i), int(j))

    components: dict[int, set[int]] = defaultdict(set)
    for fid in parent:
        components[find(fid)].add(fid)

    ordered_components = sorted(components.values(), key=lambda s: min(s))
    out: dict[int, int] = {}
    for cluster_id, members in enumerate(ordered_components):
        rep = min(
            members,
            key=lambda fid: (
                -int(global_n_fires.get(int(fid), 0)),
                int(fid),
            ),
        )
        out[cluster_id] = int(rep)
    return out


# ============================================================================
# Quality bound: cross-entropy delta proxy
# ============================================================================


def _sae_reconstruct(
    state_dict: dict[str, np.ndarray],
    residuals: np.ndarray,
) -> np.ndarray:
    """Run the SAE encode→decode loop; return the per-token
    reconstruction (shape `(n_tokens, d_model)`)."""
    pre = (residuals - state_dict["b_dec"]) @ state_dict["W_enc"] + state_dict["b_enc"]
    act = np.maximum(pre, 0.0)
    return act @ state_dict["W_dec"] + state_dict["b_dec"]


def _token_cross_entropy_delta(
    residuals: np.ndarray,
    state_before: dict[str, np.ndarray],
    state_after: dict[str, np.ndarray],
) -> float:
    """Mean per-token softmax-normalized squared distance between
    the two SAE reconstructions on the same residuals. Tractable
    proxy for proper next-token cross-entropy (would require a
    full GPT-2 forward per iteration); monotonic in actual
    reconstruction error.

    Returns a scalar in `[0, 2]` (bounded by squared L2 between two
    softmax distributions).
    """
    recon_before = _sae_reconstruct(state_before, residuals)
    recon_after = _sae_reconstruct(state_after, residuals)
    sm_before = _softmax(recon_before)
    sm_after = _softmax(recon_after)
    diff = sm_before - sm_after
    return float(np.mean(np.sum(diff * diff, axis=-1)))


def _softmax(x: np.ndarray) -> np.ndarray:
    max_x = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max_x)
    return e_x / e_x.sum(axis=-1, keepdims=True)


# ============================================================================
# Six-sigfig float rounding (matches the JSON serialization discipline so
# in-memory values round-trip exactly through to_json / from_json).
# ============================================================================


def _round_float(v: float) -> float:
    if not np.isfinite(v):
        return float(v)
    if v == 0.0:
        return 0.0
    return float(format(float(v), ".6g"))


def _round_panels(panels: list[Panel]) -> tuple[Panel, ...]:
    out: list[Panel] = []
    for p in panels:
        out.append(
            Panel(
                panel_id=p.panel_id,
                anchor=p.anchor,
                feature_ids=p.feature_ids,
                cosines_to_anchor=tuple(_round_float(c) for c in p.cosines_to_anchor),
            )
        )
    return tuple(out)
