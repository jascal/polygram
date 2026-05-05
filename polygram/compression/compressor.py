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

import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from polygram.behavioural.report import CandidatePair, ValidationReport
from polygram.compression._hash import sha256_file
from polygram.compression.report import (
    SCHEMA_VERSION,
    ClusterPlan,
    CompressionPlan,
    CompressionReport,
    CompressionResult,
)
from polygram.compression.strategies.zero import apply_zero


_SUPPORTED_STRATEGIES: frozenset[str] = frozenset({"zero"})


@dataclass
class Compressor:
    """Consumes a `ValidationReport` and rewrites an SAE checkpoint
    so the redundancies the validator confirmed become inert.

    Defaults encode the §4.4 / §5.1 calibration: the only currently
    implemented strategy is `"zero"` (see `add-compression-action/
    design.md` Decision 5); `merge` is deferred.

    `representatives` overrides the per-cluster representative pick.
    Keys are cluster ids assigned in `plan()` by ascending min-fid;
    values are feature ids that must already be members of the named
    cluster, otherwise `__post_init__` raises.
    """

    validation_report: ValidationReport
    sae_checkpoint: Path
    strategy: str = "zero"
    representatives: dict[int, int] | None = None

    # Cached union-find clusters keyed by cluster_id; populated by
    # `plan()` and consulted by the `representatives` validator.
    _cached_plan: CompressionPlan | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ----------------------------------------------------------------
    # Construction-time validation
    # ----------------------------------------------------------------

    def __post_init__(self) -> None:
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

        # Highest summed n_fires; tiebreak lowest fid.
        return min(cluster, key=lambda fid: (-n_fires_total[fid], fid))

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
        from safetensors.numpy import load_file, save_file

        from polygram.sae_import import from_sae_lens, load_sae_safetensors

        source_state = load_file(str(self.sae_checkpoint))
        source_sha = sha256_file(self.sae_checkpoint)

        rewritten = _dispatch_strategy(self.strategy, source_state, plan)

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
        )

        feature_ids = list(plan.feature_ids)
        records = load_sae_safetensors(str(out_path), feature_ids=feature_ids)
        rebuilt_dictionary, _selection_report = from_sae_lens(
            records,
            feature_ids,
            assign_gamma=True,
            name=self.validation_report.dictionary_name,
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


def _dispatch_strategy(name: str, state, plan: CompressionPlan):
    if name == "zero":
        return apply_zero(state, plan)
    raise ValueError(
        f"Compressor: unsupported strategy {name!r}; "
        f"supported: {sorted(_SUPPORTED_STRATEGIES)}"
    )
