"""SAE import utilities — bridge SAE-Lens / Anthropic-style sparse
autoencoder dictionaries into Polygram `Dictionary` objects.

The bridge is *selection-first*: real SAEs ship 16k–1M features but
Polygram's rung-1 MPS encoding holds at most 8 features per
dictionary. The user names a small subset by feature id; this module
clusters their projection vectors to assign β, surfaces fidelity stats
in a `SelectionReport`, and refuses oversized subsets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from polygram.config import SAEImportConfig  # noqa: F401
    from polygram.geometry import GeometricProfile  # noqa: F401

import numpy as np

from polygram.dictionary import Dictionary, Feature
from polygram.encoding import MPSRung1

MAX_FEATURES_PER_DICTIONARY = 8

# Decoder weight tensor key auto-detect precedence — see
# `load_sae_safetensors`. Update only with empirical signal from a
# real-world checkpoint that uses a key outside this list.
_DECODER_KEY_PRECEDENCE = ("W_dec", "decoder.weight", "dec")
_SAE_EXTRA_INSTALL_HINT = (
    "safetensors is required for load_sae_safetensors; "
    "install with `pip install polygram[sae]`."
)

# Unified alias table: source key (as it appears in the file) → canonical key.
# Covers PyTorch nn.Linear naming (LlamaScope, etc.) and legacy short forms.
_KEY_ALIASES: dict[str, str] = {
    "W_dec": "W_dec",
    "decoder.weight": "W_dec",
    "dec": "W_dec",
    "W_enc": "W_enc",
    "encoder.weight": "W_enc",
    "b_dec": "b_dec",
    "decoder.bias": "b_dec",
    "b_enc": "b_enc",
    "encoder.bias": "b_enc",
}

# Safetensors dtype string → numpy dtype (BF16 is handled specially).
_NUMPY_DTYPE_MAP: dict[str, "np.dtype[Any]"] = {
    "F32": np.dtype("float32"),
    "F16": np.dtype("float16"),
    "F64": np.dtype("float64"),
    "I8": np.dtype("int8"),
    "I16": np.dtype("int16"),
    "I32": np.dtype("int32"),
    "I64": np.dtype("int64"),
    "U8": np.dtype("uint8"),
    "U16": np.dtype("uint16"),
    "U32": np.dtype("uint32"),
    "U64": np.dtype("uint64"),
}


def _read_safetensors_header(path: Path) -> dict[str, dict]:
    """Read only the JSON metadata section from a safetensors file.

    Returns a map of tensor name → {dtype, shape, data_offsets}.
    The ``__metadata__`` key, if present, is stripped.
    """
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header_bytes = f.read(header_len)
    meta: dict[str, dict] = json.loads(header_bytes)
    meta.pop("__metadata__", None)
    return meta


def _bf16_to_f32(raw: bytes, shape: tuple) -> np.ndarray:
    """Convert raw bfloat16 bytes to a float32 ndarray.

    BF16 occupies the upper 16 bits of a float32 word; left-shifting by 16
    and reinterpreting gives the exact float32 equivalent.
    """
    u16 = np.frombuffer(raw, dtype=np.uint16)
    return (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape).copy()


def _safe_to_float32(arr: Any) -> np.ndarray:
    """Convert any safetensors-returned tensor view to a float32 ndarray.

    The safetensors slice API under ``framework="numpy"`` returns numpy
    views for native dtypes but raw objects numpy can't interpret for
    bfloat16. Future loaders are likely to hit fp8 with the same shape
    of issue. This helper centralises the conversion.
    """
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float32, copy=False)
    # Fall back to torch's float-conversion if available — handles bf16,
    # fp16, fp8, and anything else torch knows about.
    try:
        import torch  # noqa: F401

        if isinstance(arr, torch.Tensor):
            return arr.detach().to(torch.float32).cpu().numpy()
    except ImportError:
        pass
    # Last resort: try numpy's coercion. Will raise the same TypeError
    # the bf16 slice path used to raise — but with a clearer ancestor.
    return np.asarray(arr, dtype=np.float32)


def _correct_orientation(arr: np.ndarray, src_key: str) -> np.ndarray:
    """Transpose PyTorch nn.Linear weight tensors to polygram convention.

    Only acts on the two aliased keys that carry PyTorch orientation:
    - ``decoder.weight``: PyTorch (d_model, d_sae) → polygram (d_sae, d_model)
    - ``encoder.weight``: PyTorch (d_sae, d_model) → polygram (d_model, d_sae)

    Square tensors are left unchanged (orientation is ambiguous).
    """
    if arr.ndim != 2 or arr.shape[0] == arr.shape[1]:
        return arr
    if src_key == "decoder.weight":
        return arr.T
    if src_key == "encoder.weight" and arr.shape[0] > arr.shape[1]:
        return arr.T
    return arr


def _load_sae_checkpoint(
    path: Path | str,
    keys: list[str],
) -> dict[str, np.ndarray]:
    """Load named tensors from a safetensors file, normalised to float32.

    Resolves each canonical key through ``_KEY_ALIASES``, converts bfloat16
    tensors via raw-byte bit-shift (no torch), and corrects PyTorch
    ``nn.Linear`` weight orientation for ``decoder.weight`` /
    ``encoder.weight``.

    Parameters
    ----------
    path:
        Path to a ``.safetensors`` file.
    keys:
        Canonical key names to load (``"W_dec"``, ``"W_enc"``,
        ``"b_dec"``, ``"b_enc"``).

    Returns
    -------
    dict mapping each canonical key to a float32 ndarray.

    Raises
    ------
    ValueError
        When a requested canonical key has no alias present in the file.
    """
    path = Path(path)
    header = _read_safetensors_header(path)
    present = set(header.keys())

    # Resolve each canonical key to the source key actually in the file.
    src_keys: dict[str, str] = {}  # canonical → source
    for canonical in keys:
        found = next(
            (src for src, dst in _KEY_ALIASES.items() if dst == canonical and src in present),
            None,
        )
        if found is None:
            tried = sorted(src for src, dst in _KEY_ALIASES.items() if dst == canonical)
            raise ValueError(
                f"_load_sae_checkpoint: no key aliasing to {canonical!r} found in "
                f"{path}. Tried aliases {tried}; file contains: {sorted(present)}"
            )
        src_keys[canonical] = found

    if not any(header[src]["dtype"] == "BF16" for src in src_keys.values()):
        # Fast path: safetensors API handles the read directly.
        from safetensors import safe_open

        out: dict[str, np.ndarray] = {}
        with safe_open(str(path), framework="numpy") as f:
            for canonical, src in src_keys.items():
                arr = np.asarray(f.get_tensor(src), dtype=np.float32)
                out[canonical] = _correct_orientation(arr, src)
        return out

    # BF16 path: read raw tensor bytes and convert.
    with open(path, "rb") as fh:
        header_len = int.from_bytes(fh.read(8), "little")
        data_start = 8 + header_len
        out = {}
        for canonical, src in src_keys.items():
            meta = header[src]
            dtype_str: str = meta["dtype"]
            shape = tuple(meta["shape"])
            lo, hi = meta["data_offsets"]
            fh.seek(data_start + lo)
            raw = fh.read(hi - lo)
            if dtype_str == "BF16":
                arr = _bf16_to_f32(raw, shape)
            else:
                np_dt = _NUMPY_DTYPE_MAP.get(dtype_str)
                if np_dt is None:
                    raise ValueError(
                        f"_load_sae_checkpoint: unsupported dtype {dtype_str!r} "
                        f"for tensor {src!r} in {path}"
                    )
                arr = np.frombuffer(raw, dtype=np_dt).reshape(shape).astype(np.float32)
            out[canonical] = _correct_orientation(arr, src)
    return out


@dataclass(frozen=True)
class SAEFeatureRecord:
    """One feature pulled from an SAE.

    `projection` is the decoder column (or other unit-direction
    vector) for the feature in residual-stream space — what Polygram
    actually consumes.
    """

    feature_id: int
    name: str
    projection: np.ndarray
    label: str | None = None
    activation_mean: float | None = None
    activation_std: float | None = None

    def __post_init__(self) -> None:
        proj = np.asarray(self.projection, dtype=float)
        if proj.ndim != 1:
            raise ValueError(
                f"SAEFeatureRecord {self.name!r}: projection must be 1D, "
                f"got shape {proj.shape}"
            )
        if not np.all(np.isfinite(proj)):
            raise ValueError(
                f"SAEFeatureRecord {self.name!r}: projection contains "
                f"non-finite values"
            )
        # frozen dataclass — bypass to coerce dtype
        object.__setattr__(self, "projection", proj)


@dataclass(frozen=True)
class SelectionReport:
    """Fidelity stats for a `from_sae_lens(...)` call.

    `beta_variance_explained` is `1 - SS_residual / SS_total`, where
    `SS_total` is the sum of squared distances of selected projection
    vectors from their collective centroid and `SS_residual` is the
    sum of squared distances from each vector to *its assigned
    cluster's centroid*. Higher = the cluster partition captures more
    of the projection-space variance the user-selected subset carries.
    1.0 means clusters are noise-free (e.g., identical projections per
    cluster). 0.0 means the partition explains nothing.

    `reconstruction_error` is per-feature Euclidean distance from each
    projection vector to its assigned cluster centroid. `tier_preservation`
    is the Pearson correlation between off-diagonal `|G|²` entries of
    the projection-space cosine-overlap matrix and the analytic
    Polygram Gram of the built `Dictionary` at φ=0; populated only by
    the `clustered` profile (and any third-party profiles that opt to
    reuse `TierPreservationFidelity`). For other profiles
    `tier_preservation` is `None` and the profile's chosen scalar is in
    `geometric_fidelity` instead.
    `gamma_method` records `"zero"` (default) or `"projection_pca"`.
    `profile` is the name of the `GeometricProfile` used for this
    build; defaults to `"clustered"` when `from_sae_lens` is called
    without a profile. `geometric_fidelity` is the active profile's
    headline fidelity scalar; `None` when the metric isn't defined for
    this geometry / sample size.
    """

    n_input_features: int
    n_selected: int
    cluster_assignments: dict[str, str]
    cluster_method: str
    beta_variance_explained: float
    reconstruction_error: dict[str, float] = field(default_factory=dict)
    tier_preservation: float | None = None
    gamma_method: str = "zero"
    warnings: list[str] = field(default_factory=list)
    profile: str = "clustered"
    geometric_fidelity: float | None = None


def _detect_decoder_key(
    keys: list[str],
) -> tuple[str, list[str]]:
    """Return `(matched_key, sorted_present_keys)`. `matched_key` is the
    first entry of `_DECODER_KEY_PRECEDENCE` that appears in `keys`;
    raises `ValueError` listing both the precedence and what was found
    when no entry matches.
    """
    present = sorted(keys)
    for candidate in _DECODER_KEY_PRECEDENCE:
        if candidate in present:
            return candidate, present
    raise ValueError(
        f"load_sae_safetensors: no decoder weight tensor found. "
        f"Looked for {list(_DECODER_KEY_PRECEDENCE)} in priority order; "
        f"file contains: {present}"
    )


def load_sae_safetensors(
    path: str | Path,
    *,
    names: dict[int, str] | None = None,
    feature_ids: list[int] | None = None,
) -> dict[int, SAEFeatureRecord]:
    """Read a single ``.safetensors`` file and return the
    ``dict[int, SAEFeatureRecord]`` shape that
    :func:`from_sae_lens` already consumes.

    Decoder weight tensor key is auto-detected via the fixed
    precedence list ``("W_dec", "decoder.weight", "dec")``. Decoder
    rows are features (one row → one record), matching the SAE-Lens
    canonical layout. The loader transposes only when the matched key
    is ``"decoder.weight"`` and the matrix is non-square (PyTorch
    ``nn.Linear`` weight convention is ``out × in``, where for a
    decoder ``out = d_model`` and ``in = d_sae``); ``W_dec`` and
    ``dec`` are always row-as-feature.

    Returned records have ``label=None``,
    ``activation_mean=None``, and ``activation_std=None``. Attaching
    those is downstream tooling.

    Parameters
    ----------
    path : str | Path
        Path to a ``.safetensors`` file on disk.
    names : dict[int, str] | None
        Optional per-feature name override. Keys outside
        ``[0, n_features)`` raise ``ValueError``. Absent keys default
        to ``f"feat_{i}"``.
    feature_ids : list[int] | None
        When ``None`` (default), every feature is loaded eagerly.
        When set, only the named rows are read off disk via
        ``safetensors.safe_open(...).get_slice(...)`` slicing — the
        rest of the decoder tensor is never materialized in memory.
        For SAEs in the GB-class range this is the difference
        between a multi-GB working set and a few-MB one. Out-of-range
        ids raise ``ValueError``; the returned dict is keyed by
        ``feature_ids`` (so callers know they got what they asked for)
        and the dict's iteration order matches the input list.

    Raises
    ------
    ImportError
        When the optional ``[sae]`` extra is not installed (i.e. the
        ``safetensors`` package is unavailable). The message points at
        ``pip install polygram[sae]``.
    ValueError
        When the file contains no recognized decoder weight tensor,
        the matched tensor is not 2D, a ``names`` key is outside
        the valid range, or any ``feature_ids`` entry is outside
        the valid range.
    """
    try:
        from safetensors import safe_open  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch
        raise ImportError(_SAE_EXTRA_INSTALL_HINT) from exc

    if feature_ids is not None:
        return _load_subset(path, feature_ids, names=names)

    tensors = _load_sae_checkpoint(Path(path), ["W_dec"])
    weight = tensors["W_dec"]
    if weight.ndim != 2:
        raise ValueError(
            f"load_sae_safetensors: W_dec has shape "
            f"{tuple(weight.shape)}; expected 2D"
        )

    n_features = weight.shape[0]
    resolved_names: dict[int, str] = {}
    if names is not None:
        for key, value in names.items():
            if not (0 <= int(key) < n_features):
                raise ValueError(
                    f"load_sae_safetensors: names key {key!r} is outside the "
                    f"valid range [0, {n_features})"
                )
            resolved_names[int(key)] = str(value)

    out: dict[int, SAEFeatureRecord] = {}
    for i in range(n_features):
        proj = np.asarray(weight[i, :], dtype=float)
        out[i] = SAEFeatureRecord(
            feature_id=i,
            name=resolved_names.get(i, f"feat_{i}"),
            projection=proj,
            label=None,
            activation_mean=None,
            activation_std=None,
        )
    return out


def _load_subset(
    path: str | Path,
    feature_ids: list[int],
    *,
    names: dict[int, str] | None = None,
) -> dict[int, SAEFeatureRecord]:
    """Lazy-load a subset of decoder rows via ``safetensors.safe_open``.

    The full decoder tensor is never materialized in memory — each
    requested row (or column, post-orientation) is sliced individually
    off disk. For GB-class SAEs this turns a multi-GB working set into
    a per-row ``d_model × 8`` byte read.

    BF16 tensors take a separate raw-bytes path (the safetensors slice
    API cannot return bf16 to numpy). For non-transposed bf16 we read
    just the requested row's bytes directly; for the transposed case
    (PyTorch ``decoder.weight`` non-square) we materialise the full
    tensor once via the eager bf16 path. Most modern LLM SAEs use the
    fast row-slice path.
    """
    from safetensors import safe_open

    path_obj = Path(path)
    header = _read_safetensors_header(path_obj)

    with safe_open(str(path_obj), framework="numpy") as f:
        keys = list(f.keys())
        matched, _ = _detect_decoder_key(keys)
        meta = header[matched]
        shape = tuple(meta["shape"])
        dtype_str = meta["dtype"]
        if len(shape) != 2:
            raise ValueError(
                f"load_sae_safetensors: tensor {matched!r} has shape "
                f"{shape}; expected 2D"
            )
        # Mirror the eager path's orientation rule: only transpose for a
        # non-square decoder.weight (PyTorch out × in convention).
        if matched == "decoder.weight" and shape[0] != shape[1]:
            n_features = shape[1]
            transpose = True
        else:
            n_features = shape[0]
            transpose = False

        for fid in feature_ids:
            if not (0 <= int(fid) < n_features):
                raise ValueError(
                    f"load_sae_safetensors: feature_id {fid!r} is outside the "
                    f"valid range [0, {n_features})"
                )

        resolved_names: dict[int, str] = {}
        if names is not None:
            for key, value in names.items():
                if not (0 <= int(key) < n_features):
                    raise ValueError(
                        f"load_sae_safetensors: names key {key!r} is outside the "
                        f"valid range [0, {n_features})"
                    )
                resolved_names[int(key)] = str(value)

        is_bf16 = dtype_str == "BF16"
        bf16_full: np.ndarray | None = None
        if is_bf16 and transpose:
            # Transposed bf16: pull the full tensor through the eager
            # bf16 conversion. Slower than per-row but rare in practice
            # (most modern SAEs ship ``W_dec`` in (n_features, d_model)
            # layout, hitting the fast non-transpose row-slice path).
            full = _load_sae_checkpoint(path_obj, ["W_dec"])
            bf16_full = full["W_dec"]  # already float32, oriented

        slc = None if is_bf16 else f.get_slice(matched)

        out: dict[int, SAEFeatureRecord] = {}
        for fid in feature_ids:
            fid_int = int(fid)
            if is_bf16:
                if bf16_full is not None:
                    row = bf16_full[fid_int, :].astype(np.float32, copy=False)
                else:
                    # Fast path: read just this row's raw bytes.
                    lo, _ = meta["data_offsets"]
                    row_bytes = shape[1] * 2
                    with open(path_obj, "rb") as fh:
                        header_len = int.from_bytes(fh.read(8), "little")
                        data_start = 8 + header_len
                        fh.seek(data_start + lo + fid_int * row_bytes)
                        raw = fh.read(row_bytes)
                    row = _bf16_to_f32(raw, (shape[1],))
            else:
                if transpose:
                    row = _safe_to_float32(slc[:, fid_int])
                else:
                    row = _safe_to_float32(slc[fid_int, :])
            out[fid_int] = SAEFeatureRecord(
                feature_id=fid_int,
                name=resolved_names.get(fid_int, f"feat_{fid_int}"),
                projection=row,
                label=None,
                activation_mean=None,
                activation_std=None,
            )
    return out


def load_toy_sae(path: str | Path) -> dict[int, SAEFeatureRecord]:
    """Load a JSON file in the bundled toy-SAE schema.

    Schema: `{"features": [{"feature_id", "name", "projection", ...},
    ...]}`. Returns a dict keyed by `feature_id` for O(1) lookup.
    """
    p = Path(path)
    raw: dict[str, Any] = json.loads(p.read_text())
    if "features" not in raw:
        raise ValueError(f"{path}: missing top-level 'features' list")
    out: dict[int, SAEFeatureRecord] = {}
    for entry in raw["features"]:
        rec = SAEFeatureRecord(
            feature_id=int(entry["feature_id"]),
            name=str(entry["name"]),
            projection=np.asarray(entry["projection"], dtype=float),
            label=entry.get("label"),
            activation_mean=entry.get("activation_mean"),
            activation_std=entry.get("activation_std"),
        )
        if rec.feature_id in out:
            raise ValueError(
                f"{path}: duplicate feature_id {rec.feature_id}"
            )
        out[rec.feature_id] = rec
    return out


def from_sae_lens(
    records: dict[int, SAEFeatureRecord],
    feature_ids: list[int],
    *,
    name: str = "ImportedSAE",
    cluster_assignments: dict[int, str] | None = None,
    n_clusters: int | None = None,
    encoding: MPSRung1 | None = None,
    beta_range: tuple[float, float] = (-0.5, 0.5),
    assign_gamma: bool | None = None,
    gamma_range: tuple[float, float] | None = None,
    config: "SAEImportConfig | None" = None,
    profile: "str | GeometricProfile | None" = None,
) -> tuple[Dictionary, SelectionReport]:
    """Build a `Dictionary` from an explicit subset of SAE features.

    Cluster assignment precedence:

    1. `cluster_assignments` (user) — `dict[feature_id, cluster_name]`
    2. Labels of the form `"<cluster>/<name>"` — parse the prefix
    3. The active profile's `KnobAssignment` strategy (k-means or
       PCA-axis depending on profile)

    The active profile's `GeometricFidelity` is computed regardless of
    which cluster-assignment path was taken.

    Profile resolution order:
        per-field ``profile`` kwarg
        > ``SAEImportConfig.profile``
        > registry default (``"clustered"`` — v0.1.0-equivalent)

    The ``"clustered"`` profile reproduces the v0.1.0 defaults
    byte-for-byte: k=2 k-means, β = ±0.5 antipodal spread, Pearson
    `tier_preservation` fidelity. The ``"uniform-sphere"`` profile
    targets SAEs with `d_model ≥ ~1K` and `n_features ≥ ~16K` (audio
    + large LM SAEs); see ``polygram.geometry`` and
    ``docs/research/sae-geometry-regimes.md``.

    β values are spread according to the profile (`clustered`:
    cluster-ordinal antipodal; `uniform-sphere`: PCA-axis coordinate).
    α, φ default to 0. γ is per-feature PCA-derived when
    ``assign_gamma=True`` (the default). Per-field kwargs (`n_clusters`,
    `gamma_range`, `assign_gamma`, etc.) override profile defaults;
    profile defaults override strategy internal defaults. Refuses
    subsets larger than 8 features.
    """
    # Precedence: per-field kwarg (non-None) > config > profile defaults
    # > SAEImportConfig defaults. Profile is resolved at call time
    # against the live registry so v0.1.x SAEImportConfig instances
    # (no profile field) deserialise cleanly.
    from polygram.config import SAEImportConfig

    cfg = config if config is not None else SAEImportConfig()
    resolved_profile = _resolve_profile(profile, cfg)

    if assign_gamma is None:
        assign_gamma = cfg.assign_gamma
    if gamma_range is None:
        gamma_range = cfg.gamma_range
    # n_clusters default cascade: kwarg > config (only if config explicitly
    # supplied) > profile.default_n_clusters > strategy internal default.
    if n_clusters is None:
        if config is not None:
            n_clusters = cfg.n_clusters
        elif resolved_profile.default_n_clusters is not None:
            n_clusters = resolved_profile.default_n_clusters
    if len(feature_ids) > MAX_FEATURES_PER_DICTIONARY:
        raise ValueError(
            f"selected {len(feature_ids)} features, but Polygram's "
            f"rung-1 MPS encoding caps a Dictionary at "
            f"{MAX_FEATURES_PER_DICTIONARY} features. Pick a smaller "
            f"subset."
        )
    if len(feature_ids) == 0:
        raise ValueError("feature_ids is empty; nothing to import")

    missing = [fid for fid in feature_ids if fid not in records]
    if missing:
        raise ValueError(f"feature_id(s) not in records: {missing}")

    selected = [records[fid] for fid in feature_ids]
    projs = np.stack([r.projection for r in selected])
    n_features_input = len(records)

    warnings: list[str] = []

    # Cluster-assignment paths (run upstream of strategy dispatch and
    # bypass the profile's KnobAssignment). cluster_assignments and
    # from_labels are explicit user-supplied or label-derived; they are
    # not the strategy's job.
    bypass_strategy = False
    cluster_per_feature: list[str]
    method: str
    betas_explicit: list[float] | None = None
    gammas_explicit: list[float] | None = None
    var_explained_explicit: float | None = None

    if cluster_assignments is not None:
        bypass_strategy = True
        method = "user"
        for fid in feature_ids:
            if fid not in cluster_assignments:
                raise ValueError(
                    f"cluster_assignments missing entry for feature_id {fid}"
                )
        cluster_per_feature = [cluster_assignments[fid] for fid in feature_ids]
    elif all(_label_has_cluster_prefix(r.label) for r in selected):
        bypass_strategy = True
        method = "from_labels"
        cluster_per_feature = [r.label.split("/", 1)[0] for r in selected]
    else:
        # Strategy dispatch.
        n_for_warn = (
            n_clusters
            if n_clusters is not None
            else (resolved_profile.default_n_clusters or 2)
        )
        if n_for_warn > len(selected):
            warnings.append(
                f"n_clusters={n_for_warn} > selected={len(selected)}; "
                f"clamping to {len(selected)}"
            )
        result = resolved_profile.knob_assignment.assign(
            projs,
            [r.name for r in selected],
            n_clusters=n_clusters,
            gamma_range=gamma_range,
            assign_gamma=assign_gamma,
            seed=0,
        )
        cluster_per_feature = result.cluster_per_feature
        method = result.cluster_method
        betas_explicit = result.betas
        gammas_explicit = result.gammas
        var_explained_explicit = result.beta_variance_explained

    cluster_order: list[str] = []
    seen: set[str] = set()
    for c in cluster_per_feature:
        if c not in seen:
            cluster_order.append(c)
            seen.add(c)

    # When the strategy was bypassed, fall back to v0.1.0 cluster-ordinal
    # β spread + per-cluster-PCA γ + cluster-residual variance — the
    # historical contract for cluster_assignments / from_labels paths.
    if bypass_strategy:
        from polygram.geometry.clustered import (
            _centroids,
            _gamma_via_cluster_pca,
            _spread_betas,
            _variance_explained,
        )

        betas_by_cluster = _spread_betas(cluster_order, beta_range)
        centroids_by_cluster = _centroids(projs, cluster_per_feature)
        var_explained = _variance_explained(
            projs, centroids_by_cluster, cluster_per_feature
        )
        if assign_gamma:
            gammas = _gamma_via_cluster_pca(
                projs, cluster_per_feature, gamma_range
            )
            gamma_method = "projection_pca"
        else:
            gammas = [0.0] * len(selected)
            gamma_method = "zero"
        betas = [betas_by_cluster[c] for c in cluster_per_feature]
    else:
        betas = betas_explicit  # type: ignore[assignment]
        gammas = gammas_explicit  # type: ignore[assignment]
        var_explained = var_explained_explicit  # type: ignore[assignment]
        gamma_method = "projection_pca" if assign_gamma else "zero"
        # For reconstruction_error reporting we still need centroids
        # (defined as cluster mean of raw projections, regardless of
        # strategy).
        from polygram.geometry.clustered import _centroids

        centroids_by_cluster = _centroids(projs, cluster_per_feature)

    features = [
        Feature(name=r.name, cluster=c, beta=b, gamma=g)
        for r, c, b, g in zip(selected, cluster_per_feature, betas, gammas)
    ]
    hierarchy: dict[str, list[str]] = {c: [] for c in cluster_order}
    for f in features:
        hierarchy[f.cluster].append(f.name)

    dictionary = Dictionary(
        name=name,
        features=features,
        hierarchy=hierarchy,
        encoding=encoding or MPSRung1(),
    )

    reconstruction_error = {
        r.name: float(np.linalg.norm(r.projection - centroids_by_cluster[c]))
        for r, c in zip(selected, cluster_per_feature)
    }

    # Always invoke the active profile's fidelity, regardless of which
    # cluster-assignment path was taken.
    geometric_fidelity = resolved_profile.geometric_fidelity.compute(
        projs, dictionary
    )
    # tier_preservation field stays populated only for the v0.1.0
    # Pearson metric (TierPreservationFidelity, used by `clustered`).
    from polygram.geometry.clustered import TierPreservationFidelity

    if isinstance(
        resolved_profile.geometric_fidelity, TierPreservationFidelity
    ):
        tier_preservation: float | None = geometric_fidelity
    else:
        tier_preservation = None

    report = SelectionReport(
        n_input_features=n_features_input,
        n_selected=len(selected),
        cluster_assignments={r.name: c for r, c in zip(selected, cluster_per_feature)},
        cluster_method=method,
        beta_variance_explained=var_explained,
        reconstruction_error=reconstruction_error,
        tier_preservation=tier_preservation,
        gamma_method=gamma_method,
        warnings=warnings,
        profile=resolved_profile.name,
        geometric_fidelity=geometric_fidelity,
    )
    return dictionary, report


def _label_has_cluster_prefix(label: str | None) -> bool:
    return isinstance(label, str) and "/" in label and label.split("/", 1)[0]


def _resolve_profile(
    profile: "str | GeometricProfile | None", cfg: "SAEImportConfig"
) -> "GeometricProfile":
    """Resolve `from_sae_lens`'s `profile=` argument.

    Resolution order: explicit kwarg > ``cfg.profile`` (string) >
    registry default (``"clustered"``). Strings are looked up against
    the live `polygram.geometry` registry so third-party-registered
    profiles work without any plumbing here.
    """
    from polygram.geometry import (
        GeometricProfile as _GeometricProfile,
        get_profile,
    )

    if profile is None:
        cfg_profile_name = getattr(cfg, "profile", None)
        return get_profile(cfg_profile_name or "clustered")
    if isinstance(profile, str):
        return get_profile(profile)
    if isinstance(profile, _GeometricProfile):
        return profile
    raise TypeError(
        f"from_sae_lens: profile must be str | GeometricProfile | "
        f"None; got {type(profile).__name__}"
    )
