"""The `residual_kmeans` regrow strategy.

Per `add-compression-regrow/design.md` Decision 2:

1. Compute the residual stream
   `residual_stream = residuals - sae_reconstruct(residuals)`
   using the source checkpoint's tensors. Pure numpy.
2. Run `sklearn.cluster.KMeans(n_clusters=K, n_init=n_init,
   random_state=seed, algorithm='lloyd')` on the residual stream,
   where `K = len(zeroed)`.
3. Assign centroid `k` to the k-th feature id in `sorted(zeroed)`.
   Populate `W_dec[fid, :] = centroid / max(‖centroid‖, eps)`,
   `W_enc[:, fid] = W_dec[fid, :]`, `b_enc[fid] = 0`. `b_dec` is
   never modified.
4. Slots whose assigned cluster has zero residual tokens are left
   zero (not populated) and recorded in
   `RegrowReport.n_slots_left_zero`.
"""

from __future__ import annotations

import numpy as np

from polygram.compression.regrow_report import SlotPopulation


_EPS = 1e-12
_RESIDUAL_SIGNAL_FLOOR = 1e-9


def compute_residual_stream(
    state_dict: dict[str, np.ndarray],
    residuals: np.ndarray,
) -> np.ndarray:
    """Run the SAE encode→decode loop on cached residuals; return
    the per-token residual stream.

    Inputs:
      - `state_dict`: a checkpoint state-dict containing `W_enc`
        (d_model × n_features), `b_enc` (n_features,), `W_dec`
        (n_features × d_model), `b_dec` (d_model,).
      - `residuals`: cached LM residuals at the SAE's input layer,
        shape `(n_tokens, d_model)`.

    Returns the per-token residual `residuals - sae_reconstruct(...)`,
    shape `(n_tokens, d_model)`.
    """
    w_enc = state_dict["W_enc"]
    b_enc = state_dict["b_enc"]
    w_dec = state_dict["W_dec"]
    b_dec = state_dict["b_dec"]

    pre = (residuals - b_dec) @ w_enc + b_enc          # (n_tokens, n_features)
    act = np.maximum(pre, 0.0)                          # ReLU; SAE convention
    recon = act @ w_dec + b_dec                         # (n_tokens, d_model)
    return residuals - recon


def plan_kmeans(
    residual_stream: np.ndarray,
    zeroed_sorted: list[int],
    *,
    seed: int,
    n_init: int,
) -> tuple[np.ndarray, list[int]]:
    """Run k-means on the residual stream; return the cluster
    centroids (shape `(K, d_model)`) and the per-cluster size list
    (length K). The k-th centroid is assigned to the k-th feature id
    in `zeroed_sorted`.

    Raises `RuntimeError` if the residual stream has no signal
    (`std < 1e-9`); raises `ValueError` if `n_tokens < K`.
    """
    K = len(zeroed_sorted)
    if K == 0:
        return np.zeros((0, residual_stream.shape[1]), dtype=residual_stream.dtype), []

    n_tokens = int(residual_stream.shape[0])
    if n_tokens < K:
        raise ValueError(
            f"residual_kmeans: n_residual_tokens={n_tokens} is less than "
            f"the K={K} zeroed slots; use more prompts or fewer slots"
        )

    if float(np.std(residual_stream)) < _RESIDUAL_SIGNAL_FLOOR:
        raise RuntimeError(
            "residual_kmeans: residual stream has no signal "
            f"(std={float(np.std(residual_stream)):.3e} < "
            f"{_RESIDUAL_SIGNAL_FLOOR}); try a more diverse prompt set"
        )

    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=K,
        n_init=int(n_init),
        random_state=int(seed),
        algorithm="lloyd",
    )
    km.fit(residual_stream)
    labels = km.labels_                                 # (n_tokens,)
    centroids = km.cluster_centers_.astype(             # (K, d_model)
        residual_stream.dtype, copy=False
    )

    cluster_sizes = [int((labels == k).sum()) for k in range(K)]
    return centroids, cluster_sizes


def apply_residual_kmeans(
    state_dict: dict[str, np.ndarray],
    zeroed_sorted: list[int],
    centroids: np.ndarray,
    cluster_sizes: list[int],
) -> tuple[dict[str, np.ndarray], list[SlotPopulation]]:
    """Write centroids into a copy of the state-dict at the zeroed
    slots; return the rewritten state-dict and the per-slot
    diagnostics list.

    Slots whose assigned cluster has zero residual tokens are left
    zero (not populated) but still appear in the diagnostics list
    with `cluster_size = 0` and `decoder_norm / encoder_norm = 0`.
    """
    out = {k: np.array(v, copy=True) for k, v in state_dict.items()}
    w_enc = out["W_enc"]
    b_enc = out["b_enc"]
    w_dec = out["W_dec"]

    slots: list[SlotPopulation] = []
    for k, fid in enumerate(zeroed_sorted):
        cluster_size = int(cluster_sizes[k]) if k < len(cluster_sizes) else 0
        if cluster_size == 0 or k >= centroids.shape[0]:
            slots.append(
                SlotPopulation(
                    feature_id=int(fid),
                    cluster_size=cluster_size,
                    decoder_norm=0.0,
                    encoder_norm=0.0,
                )
            )
            continue

        centroid = centroids[k]
        norm = float(np.linalg.norm(centroid))
        if norm < _EPS:
            slots.append(
                SlotPopulation(
                    feature_id=int(fid),
                    cluster_size=cluster_size,
                    decoder_norm=0.0,
                    encoder_norm=0.0,
                )
            )
            continue
        unit = centroid / norm

        w_dec[fid, :] = unit
        w_enc[:, fid] = unit
        b_enc[fid] = 0.0

        # Round to 6 sigfigs to match the JSON-serialization
        # discipline; this keeps in-memory values bit-identical to
        # what round-trips through to_json/from_json.
        post_dec_norm = float(format(float(np.linalg.norm(w_dec[fid, :])), ".6g"))
        post_enc_norm = float(format(float(np.linalg.norm(w_enc[:, fid])), ".6g"))
        slots.append(
            SlotPopulation(
                feature_id=int(fid),
                cluster_size=cluster_size,
                decoder_norm=post_dec_norm,
                encoder_norm=post_enc_norm,
            )
        )

    return out, slots
