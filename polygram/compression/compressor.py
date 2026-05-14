"""`Compressor` — the loop's downstream half.

Two-stage API:

    plan() -> CompressionPlan                                    # cheap
    apply(plan=None, output_checkpoint=...) -> CompressionResult # writes
    run(output_checkpoint=...) -> CompressionResult              # both

`plan()` runs union-find on `validation_report.confirmed`, picks one
representative per connected component, and returns a deterministic
`CompressionPlan`. No I/O, no torch.

`apply()` reads the source checkpoint via `safetensors.numpy.load_file`,
applies the `zero` strategy in-memory, writes the rewritten weights
atomically to a new file (sibling temp + `os.replace`), and rebuilds a
`Dictionary` from the new checkpoint. Source bytes are never modified.

`run()` is the convenience wrapper.
"""

from __future__ import annotations

import math
import os
import tempfile
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from polygram.config import CompressionConfig  # noqa: F401

from polygram.behavioural.report import CandidatePair, ValidationReport
from polygram.compression._hash import sha256_file
from polygram.compression.report import (
    SCHEMA_VERSION,
    ClusterPlan,
    CompressionPlan,
    CompressionReport,
    CompressionResult,
)
from polygram.compression.pareto import ParetoOutcome, ParetoReport
from polygram.compression.strategies.zero import apply_zero


_SUPPORTED_STRATEGIES: frozenset[str] = frozenset({"zero", "merge"})
_SUPPORTED_REP_SELECTIONS: frozenset[str] = frozenset(
    {"n_fires", "scale_aware"}
)
_SUPPORTED_MERGE_MODES: frozenset[str] = frozenset(
    {"freq_weighted", "simple_mean"}
)


@dataclass
class Compressor:
    """Consumes a `ValidationReport` and rewrites an SAE checkpoint
    so the redundancies the validator confirmed become inert.

    Defaults encode the §4.4 / §5.1 calibration. Pass
    ``config=CompressionConfig(...)`` (see :mod:`polygram.config`) to
    swap in a typed tuning bundle; per-field kwargs win over ``config``,
    which wins over the dataclass defaults. Note that the dataclass
    defaults here (``strategy="zero"``, ``rep_selection="n_fires"``)
    differ from ``CompressionConfig``'s own iterative-friendly defaults
    (``"merge"`` / ``"scale_aware"``); a no-config call therefore
    preserves the historical defaults rather than silently switching.

    `representatives` overrides the per-cluster representative pick.
    Keys are cluster ids assigned in `plan()` by ascending min-fid;
    values are feature ids that must already be members of the named
    cluster, otherwise `__post_init__` raises.
    """

    validation_report: ValidationReport
    sae_checkpoint: Path
    # Tuning fields default to ``None`` as a sentinel; ``__post_init__``
    # resolves them via per-field-kwarg > config > legacy-default. Note
    # that the legacy Compressor defaults (``"zero"`` / ``"n_fires"`` /
    # ``"freq_weighted"``) differ from ``CompressionConfig``'s own
    # iterative-friendly defaults (``"merge"`` / ``"scale_aware"`` /
    # ``"freq_weighted"``); a no-config call therefore preserves the
    # historical Compressor defaults rather than silently switching to
    # the config defaults. See :mod:`polygram.config`.
    strategy: str | None = None
    rep_selection: str | None = None
    merge_mode: str | None = None
    representatives: dict[int, int] | None = None
    config: "CompressionConfig | None" = None
    # Encoding-aware-knob-assignment plumbing. When set, Compressor.apply
    # passes them through to its post-compression `from_sae_lens` rebuild
    # so the resulting Dictionary uses the configured encoding's full
    # state space rather than collapsing to MPSRung1-equivalent. Default
    # values preserve byte-identity.
    encoding: object | None = None
    assign_amp_knobs: bool = False

    # Cached union-find clusters keyed by cluster_id; populated by
    # `plan()` and consulted by the `representatives` validator.
    _cached_plan: CompressionPlan | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # Cached W_dec rows for `rep_selection="scale_aware"`; loaded once
    # in `_build_plan` and re-used in `apply()` to avoid a second read.
    _cached_w_dec: np.ndarray | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ----------------------------------------------------------------
    # Construction-time validation
    # ----------------------------------------------------------------

    def __post_init__(self) -> None:
        # Precedence: per-field kwarg (already set on instance) > config
        # > legacy-default. Resolve before the existing range checks.
        if self.config is not None:
            if self.strategy is None:
                self.strategy = self.config.strategy
            if self.rep_selection is None:
                self.rep_selection = self.config.rep_selection
            if self.merge_mode is None:
                self.merge_mode = self.config.merge_mode
        # Legacy fallbacks (preserves pre-config-rewrite behaviour for
        # the no-config-no-kwargs construction path).
        if self.strategy is None:
            self.strategy = "zero"
        if self.rep_selection is None:
            self.rep_selection = "n_fires"
        if self.merge_mode is None:
            self.merge_mode = "freq_weighted"

        self.sae_checkpoint = Path(self.sae_checkpoint)
        if not self.sae_checkpoint.is_file():
            raise ValueError(
                f"Compressor: --sae-checkpoint not found: "
                f"{self.sae_checkpoint}"
            )
        if self.strategy not in _SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Compressor: unsupported strategy {self.strategy!r}; "
                f"supported: {sorted(_SUPPORTED_STRATEGIES)}"
            )
        if self.rep_selection not in _SUPPORTED_REP_SELECTIONS:
            raise ValueError(
                f"Compressor: unsupported rep_selection "
                f"{self.rep_selection!r}; "
                f"supported: {sorted(_SUPPORTED_REP_SELECTIONS)}"
            )
        if self.merge_mode not in _SUPPORTED_MERGE_MODES:
            raise ValueError(
                f"Compressor: unsupported merge_mode {self.merge_mode!r}; "
                f"supported: {sorted(_SUPPORTED_MERGE_MODES)}"
            )
        if self.representatives is not None:
            self._validate_representatives_against_plan()

    def _validate_representatives_against_plan(self) -> None:
        """Build a tentative plan (without overrides) and verify every
        override key names a real cluster_id and every override value
        is a member of that cluster."""
        assert self.representatives is not None
        # Build the default plan once and cache it; `plan()` will reuse.
        default_plan = self._build_plan(apply_overrides=False)
        cluster_by_id = {c.cluster_id: c for c in default_plan.clusters}
        for cid, fid in self.representatives.items():
            if cid not in cluster_by_id:
                raise ValueError(
                    f"Compressor: representatives override names "
                    f"cluster_id={cid} which does not exist in the plan "
                    f"(known: {sorted(cluster_by_id)})"
                )
            if fid not in cluster_by_id[cid].members:
                raise ValueError(
                    f"Compressor: representatives[{cid}]={fid} is not a "
                    f"member of cluster {cid} "
                    f"(members: {list(cluster_by_id[cid].members)})"
                )

    # ----------------------------------------------------------------
    # plan()
    # ----------------------------------------------------------------

    def plan(self) -> CompressionPlan:
        """Build clusters from `validation_report.confirmed`; pick
        representatives. Idempotent / deterministic.
        """
        if self._cached_plan is not None:
            return self._cached_plan
        out = self._build_plan(apply_overrides=True)
        object.__setattr__(self, "_cached_plan", out)
        return out

    def _build_plan(self, *, apply_overrides: bool) -> CompressionPlan:
        confirmed = self.validation_report.confirmed
        # For scale_aware rep_selection we need W_dec norms; load once
        # and cache for reuse in `apply()`.
        # ~36 MB for a 4k-feature SAE @ float32 — acceptable cost for
        # scale_aware. (See design.md performance note on lazy caching
        # if this ever becomes a bottleneck on huge SAEs.)
        if (
            self.rep_selection == "scale_aware"
            and self._cached_w_dec is None
        ):
            from polygram.sae_import import _load_sae_checkpoint

            state = _load_sae_checkpoint(self.sae_checkpoint, ["W_dec"])
            object.__setattr__(self, "_cached_w_dec", state["W_dec"])

        # Union-Find on confirmed-pair endpoints.
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
            # Smaller root wins, for deterministic component leaders.
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

        for i, j in confirmed:
            for fid in (i, j):
                if fid not in parent:
                    parent[fid] = fid
            union(i, j)

        components: dict[int, set[int]] = defaultdict(set)
        for fid in parent:
            components[find(fid)].add(fid)

        # Index pairs by frozenset for n_fires lookup.
        pair_lookup: dict[tuple[int, int], CandidatePair] = {}
        for p in self.validation_report.pairs:
            key = (min(p.i, p.j), max(p.i, p.j))
            pair_lookup[key] = p

        # Materialize ClusterPlan objects in ascending min-fid order.
        ordered_components = sorted(
            components.values(), key=lambda s: min(s)
        )
        cluster_plans: list[ClusterPlan] = []
        for cluster_id, members_set in enumerate(ordered_components):
            members_sorted = tuple(sorted(members_set))
            rep = self._pick_representative(
                cluster=members_set,
                cluster_id=cluster_id,
                pair_lookup=pair_lookup,
                apply_overrides=apply_overrides,
            )
            zeroed = tuple(fid for fid in members_sorted if fid != rep)
            cluster_plans.append(
                ClusterPlan(
                    cluster_id=cluster_id,
                    members=members_sorted,
                    representative=rep,
                    zeroed=zeroed,
                )
            )

        return CompressionPlan(
            clusters=tuple(cluster_plans),
            feature_ids=tuple(int(f) for f in self.validation_report.feature_ids),
        )

    def _pick_representative(
        self,
        *,
        cluster: set[int],
        cluster_id: int,
        pair_lookup: dict[tuple[int, int], CandidatePair],
        apply_overrides: bool,
    ) -> int:
        if (
            apply_overrides
            and self.representatives is not None
            and cluster_id in self.representatives
        ):
            return self.representatives[cluster_id]

        n_fires_total: dict[int, int] = defaultdict(int)
        for (a, b), pair in pair_lookup.items():
            if a in cluster and b in cluster:
                n_fires_total[a] += pair.n_fires_i
                n_fires_total[b] += pair.n_fires_j

        if self.rep_selection == "scale_aware":
            assert self._cached_w_dec is not None
            return _score_scale_aware(
                cluster=cluster,
                pair_lookup=pair_lookup,
                w_dec=self._cached_w_dec,
                n_fires_total=n_fires_total,
            )

        # Default: highest summed n_fires; tiebreak lowest fid.
        return min(cluster, key=lambda fid: (-n_fires_total[fid], fid))

    # ----------------------------------------------------------------
    # plan_with_target() — target-K compression
    # ----------------------------------------------------------------

    def plan_with_target(
        self, target_n_features_kept: int | None = None
    ) -> CompressionPlan:
        """Build clusters by greedy union-find over score-sorted pairs
        until the cluster count first drops to `<= target_n_features_kept`.

        Ignores `validation_report.confirmed`; orders the full
        `validation_report.pairs` list by `config.score_field` descending
        and unions greedily. The cluster-count trajectory typically grows
        from 0 to a peak (as disjoint pairs introduce new components)
        then shrinks (as later pairs bridge components); the stop
        condition fires the first time count drops to `<= target_k`
        AFTER previously exceeding it. If count never exceeds target_k,
        the full pair list is processed and the most-compressed reachable
        plan is returned (caller detects via `plan.n_features_kept`).

        See `openspec/changes/add-pareto-target-compression/design.md`
        Decision 10 and `tasks.md` §3.2 for the infeasible-target
        behaviour and the stop-rule clarification.
        """
        if target_n_features_kept is None:
            if (
                self.config is None
                or self.config.target_n_features_kept is None
            ):
                raise ValueError(
                    "Compressor.plan_with_target: target_n_features_kept "
                    "must be provided either as the method argument or "
                    "via CompressionConfig(target_n_features_kept=...). "
                    "Both are None."
                )
            target_n_features_kept = self.config.target_n_features_kept

        if (
            not isinstance(target_n_features_kept, int)
            or isinstance(target_n_features_kept, bool)
            or target_n_features_kept < 1
        ):
            raise ValueError(
                f"Compressor.plan_with_target: target_n_features_kept "
                f"must be an integer >= 1; got {target_n_features_kept!r}"
            )

        score_field = (
            self.config.score_field
            if self.config is not None
            else "polygram_overlap"
        )

        # Lazy-load W_dec for scale_aware rep_selection (mirrors _build_plan).
        if (
            self.rep_selection == "scale_aware"
            and self._cached_w_dec is None
        ):
            from polygram.sae_import import _load_sae_checkpoint

            state = _load_sae_checkpoint(self.sae_checkpoint, ["W_dec"])
            object.__setattr__(self, "_cached_w_dec", state["W_dec"])

        sorted_pairs, pair_lookup = self._prepare_pairs_for_target_k(
            score_field
        )

        return self._greedy_union_to_target(
            target_n_features_kept, sorted_pairs, pair_lookup
        )

    def _prepare_pairs_for_target_k(
        self, score_field: str
    ) -> tuple[list[CandidatePair], dict[tuple[int, int], CandidatePair]]:
        """Filter + sort the pair list for target-K planning, and build
        the canonical `(min, max)` pair lookup used by
        `_pick_representative`. Shared between `plan_with_target` and
        `plan_pareto` so a multi-K sweep sorts pairs exactly once.
        """
        filtered = self._filter_pairs_for_score(
            self.validation_report.pairs, score_field
        )
        sorted_pairs = self._sort_pairs_by_score(filtered, score_field)
        pair_lookup: dict[tuple[int, int], CandidatePair] = {}
        for p in self.validation_report.pairs:
            key = (min(p.i, p.j), max(p.i, p.j))
            pair_lookup[key] = p
        return sorted_pairs, pair_lookup

    @staticmethod
    def _sort_pairs_by_score(
        pairs: tuple[CandidatePair, ...] | list[CandidatePair],
        score_field: str,
    ) -> list[CandidatePair]:
        """Decision 6: sort by `(-score, min(i,j), max(i,j))` for a
        symmetric, deterministic tiebreak on the three bounded
        score axes. Extracted so `plan_pareto` can patch / spy on it
        to verify the single-sort invariant.
        """
        return sorted(
            pairs,
            key=lambda p: (
                -float(getattr(p, score_field)),
                min(p.i, p.j),
                max(p.i, p.j),
            ),
        )

    @staticmethod
    def _filter_pairs_for_score(
        pairs: tuple[CandidatePair, ...],
        score_field: str,
    ) -> tuple[CandidatePair, ...]:
        """Drop pairs whose `score_field` value is NaN.

        Decision 5: a decoder-only `ValidationReport` (from
        `DecoderGeometryConfirmer`) populates only `decoder_overlap` and
        leaves the behavioural fields as NaN. Sorting NaN values is
        platform-dependent, so we filter first and raise if no pairs
        survive — naming the score field and the likely cause so the
        caller picks a compatible field.
        """
        filtered = tuple(
            p for p in pairs
            if not math.isnan(float(getattr(p, score_field)))
        )
        if not filtered:
            raise ValueError(
                "Compressor.plan_with_target: no pairs have a finite "
                f"{score_field!r} score after NaN filtering. Likely "
                "cause: the ValidationReport was produced by "
                "DecoderGeometryConfirmer (which populates only "
                "decoder_overlap) but the chosen score_field requires "
                "behavioural scoring. Re-validate via BehaviouralValidator "
                "or pass score_field='decoder_overlap'."
            )
        return filtered

    def _greedy_union_to_target(
        self,
        target_k: int,
        sorted_pairs: list[CandidatePair],
        pair_lookup: dict[tuple[int, int], CandidatePair],
    ) -> CompressionPlan:
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        n_clusters = 0
        target_was_exceeded = False

        for pair in sorted_pairs:
            i, j = int(pair.i), int(pair.j)
            i_new = i not in parent
            j_new = j not in parent
            if i_new:
                parent[i] = i
            if j_new:
                parent[j] = j

            ri = find(i)
            rj = find(j)
            if ri != rj:
                if i_new and j_new:
                    n_clusters += 1
                elif (not i_new) and (not j_new):
                    n_clusters -= 1
                # else (exactly one new): a singleton joins an existing
                # ≥2-member component — no change in cluster count.

                if ri < rj:
                    parent[rj] = ri
                else:
                    parent[ri] = rj

            if n_clusters > target_k:
                target_was_exceeded = True
            elif target_was_exceeded:
                # Crossed back to ≤ target_k from above — stop.
                break

        return self._materialise_plan_from_parent(parent, pair_lookup)

    def _materialise_plan_from_parent(
        self,
        parent: dict[int, int],
        pair_lookup: dict[tuple[int, int], CandidatePair],
    ) -> CompressionPlan:
        """Build a `CompressionPlan` from a union-find `parent` map.

        Rebuilds components from `parent` rather than maintaining a
        parallel membership dict inside the hot path: the union-find
        walk only needs find() roots and the `n_clusters` counter; full
        membership is only consulted here to materialise `ClusterPlan`
        objects. Two passes over `parent` (one for find(), one for
        grouping) is cheaper than synchronising membership state on
        every union.

        Shared by `_greedy_union_to_target` (single-K) and `plan_pareto`
        (multi-K via per-K parent snapshots).
        """

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        components: dict[int, set[int]] = defaultdict(set)
        for fid in list(parent):
            components[find(fid)].add(fid)

        ordered_components = sorted(
            components.values(), key=lambda s: min(s)
        )
        cluster_plans: list[ClusterPlan] = []
        for cluster_id, members_set in enumerate(ordered_components):
            members_sorted = tuple(sorted(members_set))
            rep = self._pick_representative(
                cluster=members_set,
                cluster_id=cluster_id,
                pair_lookup=pair_lookup,
                apply_overrides=True,
            )
            zeroed = tuple(fid for fid in members_sorted if fid != rep)
            cluster_plans.append(
                ClusterPlan(
                    cluster_id=cluster_id,
                    members=members_sorted,
                    representative=rep,
                    zeroed=zeroed,
                )
            )

        return CompressionPlan(
            clusters=tuple(cluster_plans),
            feature_ids=tuple(
                int(f) for f in self.validation_report.feature_ids
            ),
        )

    # ----------------------------------------------------------------
    # plan_pareto() — multi-K sweep, one sort + one union-find walk
    # ----------------------------------------------------------------

    def plan_pareto(self, targets) -> ParetoReport:
        """Build a `CompressionPlan` for each target K in one shared
        sort + one shared union-find walk.

        `targets` is any sequence of positive integers. The returned
        `ParetoReport.targets` is deduplicated and sorted descending
        for stable iteration; `outcomes[i]` matches `targets[i]`.

        Algorithm: walk the score-sorted pair list once, tracking the
        cluster-count trajectory. For each K (sorted descending), the
        stop condition fires the first time the count drops back to
        `<= K` after exceeding it (Phase 1 stop-rule clarification —
        see `_greedy_union_to_target` and openspec Decision 10).
        Snapshot the `parent` dict at each K's stop point so the plan
        for that K can be materialised independently. K values whose
        trajectory never exceeds their threshold (target too high) or
        never drops back (target too low / infeasible) take the final
        `parent` state.
        """
        if targets is None:
            raise ValueError(
                "Compressor.plan_pareto: targets must be a non-empty "
                "sequence of positive integers; got None."
            )

        target_set: set[int] = set()
        for t in targets:
            if (
                not isinstance(t, int)
                or isinstance(t, bool)
                or t < 1
            ):
                raise ValueError(
                    "Compressor.plan_pareto: every target must be an "
                    f"integer >= 1; got {t!r} in {list(targets)!r}"
                )
            target_set.add(int(t))
        if not target_set:
            raise ValueError(
                "Compressor.plan_pareto: targets is empty; supply at "
                "least one positive integer K."
            )
        sorted_targets: tuple[int, ...] = tuple(
            sorted(target_set, reverse=True)
        )

        score_field = (
            self.config.score_field
            if self.config is not None
            else "polygram_overlap"
        )

        if (
            self.rep_selection == "scale_aware"
            and self._cached_w_dec is None
        ):
            from polygram.sae_import import _load_sae_checkpoint

            state = _load_sae_checkpoint(self.sae_checkpoint, ["W_dec"])
            object.__setattr__(self, "_cached_w_dec", state["W_dec"])

        sorted_pairs, pair_lookup = self._prepare_pairs_for_target_k(
            score_field
        )

        snapshots = self._walk_pairs_for_pareto_snapshots(
            sorted_pairs, sorted_targets
        )

        outcomes: list[ParetoOutcome] = []
        for k in sorted_targets:
            plan = self._materialise_plan_from_parent(
                snapshots[k], pair_lookup
            )
            outcomes.append(
                ParetoOutcome(
                    target_k=k,
                    reached_target=plan.n_features_kept <= k,
                    plan=plan,
                )
            )

        return ParetoReport(
            schema_version=1,
            sae_checkpoint=Path(self.sae_checkpoint),
            sae_checkpoint_sha256=sha256_file(self.sae_checkpoint),
            score_field=score_field,
            targets=sorted_targets,
            outcomes=tuple(outcomes),
        )

    def _walk_pairs_for_pareto_snapshots(
        self,
        sorted_pairs: list[CandidatePair],
        sorted_targets: tuple[int, ...],
    ) -> dict[int, dict[int, int]]:
        """Single union-find walk that snapshots `parent` for each K
        as its stop condition fires. Caller materialises plans from the
        snapshots via `_materialise_plan_from_parent`.

        K values whose trajectory never exceeds K (target too high) or
        never drops back to K from above (target too low / infeasible)
        get the final `parent` state, matching Phase 1's behaviour.
        """
        parent: dict[int, int] = {}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        n_clusters = 0
        target_was_exceeded: dict[int, bool] = {k: False for k in sorted_targets}
        snapshots: dict[int, dict[int, int]] = {}

        for pair in sorted_pairs:
            i, j = int(pair.i), int(pair.j)
            i_new = i not in parent
            j_new = j not in parent
            if i_new:
                parent[i] = i
            if j_new:
                parent[j] = j

            ri = find(i)
            rj = find(j)
            if ri != rj:
                if i_new and j_new:
                    n_clusters += 1
                elif (not i_new) and (not j_new):
                    n_clusters -= 1
                if ri < rj:
                    parent[rj] = ri
                else:
                    parent[ri] = rj

            # Check each target's stop condition after this pair. Larger
            # K thresholds fire earlier in the trajectory; smaller K
            # thresholds fire later. Iterate in the order the caller
            # supplied (descending) so the first K to snapshot is the
            # largest still-unresolved target.
            for k in sorted_targets:
                if k in snapshots:
                    continue
                if n_clusters > k:
                    target_was_exceeded[k] = True
                elif target_was_exceeded[k]:
                    snapshots[k] = dict(parent)

            if len(snapshots) == len(sorted_targets):
                break

        # K values that never reached a stop condition fall back to the
        # final parent state. `dict(parent)` snapshots once per K to keep
        # them independent if a caller wanted to mutate later (defensive).
        final_parent = dict(parent)
        for k in sorted_targets:
            if k not in snapshots:
                snapshots[k] = dict(final_parent)

        return snapshots

    # ----------------------------------------------------------------
    # apply()
    # ----------------------------------------------------------------

    def apply(
        self,
        plan: CompressionPlan | None = None,
        output_checkpoint: str | os.PathLike | None = None,
    ) -> CompressionResult:
        if output_checkpoint is None:
            raise ValueError(
                "Compressor.apply: output_checkpoint is required"
            )
        if plan is None:
            plan = self.plan()

        out_path = Path(output_checkpoint).resolve()
        if out_path == self.sae_checkpoint.resolve():
            raise ValueError(
                f"Compressor.apply: output_checkpoint must differ from "
                f"the source checkpoint (both resolved to {out_path})"
            )

        # Lazy imports — keep the module import-cheap.
        from safetensors.numpy import save_file

        from polygram.sae_import import (
            _load_sae_checkpoint,
            from_sae_lens,
            load_sae_safetensors,
        )

        source_state = _load_sae_checkpoint(
            self.sae_checkpoint, ["W_dec", "W_enc", "b_dec", "b_enc"]
        )
        source_sha = sha256_file(self.sae_checkpoint)

        # Compute scale stats from the source W_dec, then apply strategy.
        cluster_norm_stats = _compute_cluster_norm_stats(
            source_state["W_dec"], plan
        )
        n_fires_by_fid = _aggregate_n_fires(self.validation_report)
        rewritten, merged_norms = _dispatch_strategy(
            self.strategy,
            source_state,
            plan,
            merge_mode=self.merge_mode,
            n_fires_by_fid=n_fires_by_fid,
        )
        plan = _patch_cluster_scale_fields(
            plan, cluster_norm_stats, merged_norms
        )
        scale_compression_ratio = _compute_scale_compression_ratio(
            source_state["W_dec"], plan, merged_norms
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: temp file in the same directory, then os.replace.
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

        n_zeroed = sum(len(c.zeroed) for c in plan.clusters)
        n_kept = sum(1 for _ in plan.clusters)
        report = CompressionReport(
            schema_version=SCHEMA_VERSION,
            source_checkpoint=str(self.sae_checkpoint),
            source_checkpoint_sha256=source_sha,
            output_checkpoint=str(out_path),
            output_checkpoint_sha256=output_sha,
            validation_report_dictionary_name=(
                self.validation_report.dictionary_name
            ),
            validation_report_schema_version=int(
                self.validation_report.schema_version
            ),
            strategy=self.strategy,
            plan=plan,
            n_features_zeroed=n_zeroed,
            n_features_kept=n_kept,
            n_clusters=len(plan.clusters),
            scale_compression_ratio=scale_compression_ratio,
        )

        # MPSRung1 caps a Dictionary at `MPSRung1.max_features` (= 8).
        # For single-panel use, plan.feature_ids ≤ cap by validator
        # contract. For the EpochCompressor's synthetic multi-panel
        # report, the union of every panel's feature_ids can be much
        # larger; cap the Dictionary rebuild to the first
        # `MPSRung1.max_features` ids. The zeroing already happened
        # on the full plan.clusters[*].zeroed list — the rebuilt
        # Dictionary is a debugging aid covering the lowest-fid
        # `cap` features, not a complete enumeration of the SAE.
        # When Compressor gains a configurable encoding (issue #48),
        # this should query the supplied encoding's max_features
        # rather than hardcoding MPSRung1.
        from polygram.encoding import MPSRung1 as _MPSRung1

        rebuild_cap = int(_MPSRung1.max_features)
        feature_ids = list(
            plan.feature_ids[: min(rebuild_cap, len(plan.feature_ids))]
        )
        records = load_sae_safetensors(str(out_path), feature_ids=feature_ids)
        rebuilt_dictionary, _selection_report = from_sae_lens(
            records,
            feature_ids,
            assign_gamma=True,
            name=self.validation_report.dictionary_name,
            encoding=self.encoding,
            assign_amp_knobs=self.assign_amp_knobs,
        )

        return CompressionResult(
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
    ) -> CompressionResult:
        return self.apply(self.plan(), output_checkpoint=output_checkpoint)


# ============================================================================
# Helpers
# ============================================================================


def _dispatch_strategy(
    name: str,
    state,
    plan: CompressionPlan,
    *,
    merge_mode: str = "freq_weighted",
    n_fires_by_fid: dict[int, int] | None = None,
):
    if name == "zero":
        return apply_zero(state, plan), None
    if name == "merge":
        from polygram.compression.strategies.merge import apply_merge

        return apply_merge(
            state,
            plan,
            merge_mode=merge_mode,
            n_fires_by_fid=n_fires_by_fid,
        )
    raise ValueError(
        f"Compressor: unsupported strategy {name!r}; "
        f"supported: {sorted(_SUPPORTED_STRATEGIES)}"
    )


# ============================================================================
# scale_aware rep selection
# ============================================================================


_NORM_EPS = 1e-8


def _normalise_minmax(values: np.ndarray) -> np.ndarray:
    """Min-max scale to [0, 1]; constant input → all zeros."""
    lo, hi = float(values.min()), float(values.max())
    if hi - lo < _NORM_EPS:
        return np.zeros_like(values)
    return (values - lo) / (hi - lo)


def _score_scale_aware(
    *,
    cluster: set[int],
    pair_lookup: dict[tuple[int, int], CandidatePair],
    w_dec: np.ndarray,
    n_fires_total: dict[int, int],
) -> int:
    """Score each candidate in `cluster` and return the best fid
    (tiebreak: lowest fid). Score = 0.4·norm_proximity + 0.4·ablation
    + 0.2·log_freq, all min-max normalised across the cluster.

    If every kl_ablate value for this cluster is NaN (geometry-only
    confirmer), the ablation term is zeroed and a UserWarning is
    emitted once per call.
    """
    members = sorted(cluster)
    norms = np.array(
        [float(np.linalg.norm(w_dec[fid])) for fid in members],
        dtype=np.float64,
    )
    median_norm = float(np.median(norms))
    norm_proximity = 1.0 - np.abs(norms - median_norm) / (
        median_norm + _NORM_EPS
    )
    norm_proximity = np.clip(norm_proximity, 0.0, 1.0)

    # Ablation importance: sum kl_ablate over pairs touching each fid.
    ablation_raw: dict[int, float] = {fid: 0.0 for fid in members}
    ablation_seen: dict[int, bool] = {fid: False for fid in members}
    for (a, b), pair in pair_lookup.items():
        if a in cluster and b in cluster:
            if not math.isnan(pair.kl_ablate_i):
                ablation_raw[a] += float(pair.kl_ablate_i)
                ablation_seen[a] = True
            if not math.isnan(pair.kl_ablate_j):
                ablation_raw[b] += float(pair.kl_ablate_j)
                ablation_seen[b] = True
    if not any(ablation_seen.values()):
        warnings.warn(
            "scale_aware rep_selection: kl_ablate is NaN for every "
            "pair in cluster; falling back to n_fires-only scoring",
            UserWarning,
            stacklevel=3,
        )
        ablation_norm = np.zeros(len(members), dtype=np.float64)
    else:
        ablation_norm = _normalise_minmax(
            np.array([ablation_raw[fid] for fid in members], dtype=np.float64)
        )

    log_freq = np.log(
        np.array(
            [float(n_fires_total[fid]) + _NORM_EPS for fid in members],
            dtype=np.float64,
        )
    )
    log_freq_norm = _normalise_minmax(log_freq)

    score = 0.4 * norm_proximity + 0.4 * ablation_norm + 0.2 * log_freq_norm
    # argmax on (score, -fid) so highest score wins; tiebreak lowest fid.
    best_idx = int(
        max(range(len(members)), key=lambda i: (score[i], -members[i]))
    )
    return members[best_idx]


# ============================================================================
# Per-cluster scale statistics
# ============================================================================


def _compute_cluster_norm_stats(
    w_dec: np.ndarray, plan: CompressionPlan
) -> dict[int, tuple[float, float]]:
    """Return cluster_id → (norm_mean, norm_std) over members."""
    out: dict[int, tuple[float, float]] = {}
    for cluster in plan.clusters:
        norms = np.linalg.norm(w_dec[list(cluster.members), :], axis=1)
        out[cluster.cluster_id] = (
            float(norms.mean()),
            float(norms.std()),
        )
    return out


def _patch_cluster_scale_fields(
    plan: CompressionPlan,
    cluster_norm_stats: dict[int, tuple[float, float]],
    merged_norms: dict[int, float] | None,
) -> CompressionPlan:
    """Rebuild a CompressionPlan with cluster_norm_mean / std / merged_norm
    populated on each ClusterPlan.
    """
    new_clusters = []
    for c in plan.clusters:
        mean_, std_ = cluster_norm_stats.get(c.cluster_id, (None, None))
        merged = (
            None if merged_norms is None else merged_norms.get(c.cluster_id)
        )
        new_clusters.append(
            ClusterPlan(
                cluster_id=c.cluster_id,
                members=c.members,
                representative=c.representative,
                zeroed=c.zeroed,
                cluster_norm_mean=mean_,
                cluster_norm_std=std_,
                merged_norm=merged,
            )
        )
    return CompressionPlan(
        clusters=tuple(new_clusters),
        feature_ids=plan.feature_ids,
    )


def _compute_scale_compression_ratio(
    w_dec_source: np.ndarray,
    plan: CompressionPlan,
    merged_norms: dict[int, float] | None,
) -> float:
    """Total preserved norm mass / total source norm mass.

    For ``zero``: preserved = rep_norm_before per cluster. For
    ``merge``: preserved = merged_norm × cluster_size — i.e., the
    rep's rescaled row "stands in" for every member it absorbed.
    Under ``simple_mean`` this equals the source sum exactly, so
    the ratio is 1.0; under ``freq_weighted`` it depends on how
    fires correlate with norms. Returns 1.0 if there are no
    clusters.

    Note: only cluster members participate in this ratio.
    Singleton (un-clustered) features are excluded from both
    numerator and denominator — see the open question in
    ``openspec/changes/scale-aware-compression/design.md``.
    Their norms are preserved exactly by the strategy, so
    counting them would just bias the ratio toward 1.0 in a
    way that hides cluster-level loss.
    """
    if not plan.clusters:
        return 1.0
    total_before = 0.0
    total_after = 0.0
    for c in plan.clusters:
        norms = np.linalg.norm(w_dec_source[list(c.members), :], axis=1)
        total_before += float(norms.sum())
        if merged_norms is not None and c.cluster_id in merged_norms:
            total_after += float(merged_norms[c.cluster_id]) * len(c.members)
        else:
            total_after += float(
                np.linalg.norm(w_dec_source[c.representative])
            )
    if total_before <= 0.0:
        return 1.0
    return total_after / total_before


def _aggregate_n_fires(report: ValidationReport) -> dict[int, int]:
    """Sum n_fires_i / n_fires_j across every pair to get a per-fid
    activation-count proxy."""
    out: dict[int, int] = defaultdict(int)
    for p in report.pairs:
        out[p.i] += int(p.n_fires_i)
        out[p.j] += int(p.n_fires_j)
    return out
