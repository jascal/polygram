"""Real-SAE A/B comparison of uniform vs partitioned polygram
compression — heuristic-driven partition (not hand-coded), real
geometry-based ValidationReport.

Pipeline:
  1. Slice jbloom GPT-2 SAE to 64 features (CPU-friendly; cached from
     PR #69's §8.4 smoke).
  2. Build SAEFeatureRecord dict + run DecoderGeometryConfirmer →
     ValidationReport (decoder-cosine²; no model forward — fast).
  3. Compute heaviness score per feature: `decoder_norm² × pair_count`
     where pair_count is the number of confirmed pairs the feature
     appears in. Higher → likely more "load-bearing" (involved in
     redundancy clusters, has substantial decoder norm).
  4. Pick top-K (K=4) by heaviness → heavy block (Rung5).
     Rest → tail block (MPSRung1).
  5. Run polygram Compressor.apply twice on the SAME SAE +
     ValidationReport:
       Run A: no partition (uniform path, encoding=Rung5 implicit
              via Compressor's encoding= field).
       Run B: partition=(heavy, tail).
  6. Compare:
       - n_features_kept / n_features_zeroed / n_clusters
       - scale_compression_ratio (top + per-block)
       - rank_ratio / post_A (per-block when available)
       - substrate cost (slot count × n_features_kept per block)
  7. Run sae-forge ForgePipeline.run_synthetic against both compressed
     outputs (if sae-forge is available and the SAE/host is compatible).
     Compare faithfulness_kl.

This is the "real production payoff" experiment for
add-encoding-partition, separate from the substrate-cost-only synth
measurement in PR #111.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Resolve repo paths relative to this file so the script is portable across
# checkouts and machines (was hard-coded to a macOS home dir). polygram is this
# repo's root (runs/ -> polygram/); sae-forge is an optional sibling repo whose
# location can be overridden with the SAE_FORGE_ROOT env var.
_POLYGRAM_ROOT = Path(__file__).resolve().parents[1]
_SAE_FORGE_ROOT = Path(os.environ.get("SAE_FORGE_ROOT", _POLYGRAM_ROOT.parent / "sae-forge"))
sys.path.insert(0, str(_POLYGRAM_ROOT))
sys.path.insert(0, str(_SAE_FORGE_ROOT / "examples"))

import numpy as np


SAE_REPO = "jbloom/GPT2-Small-SAEs-Reformatted"
SAE_FILE = "blocks.8.hook_resid_pre/sae_weights.safetensors"
N_FEATURES = 64  # slice size; budget-friendly
HEAVY_K = 4      # how many features go in the heavy block


def _slot_cost(encoding_class: str, encoding_kwargs: dict) -> int:
    if encoding_class == "Rung5":
        n = int(encoding_kwargs.get("n_amp_qubits", 4))
        return 8 * (2 ** n)
    if encoding_class == "HEA_Rung2":
        n = int(encoding_kwargs.get("n_qubits", 6))
        return 2 ** n
    return {"MPSRung1": 8, "Rung3": 16, "Rung4": 32}.get(encoding_class, 0)


def main():
    from forge_gpt2_real_sae import slice_sae_to_features  # type: ignore
    from huggingface_hub import hf_hub_download
    from safetensors.numpy import load_file, save_file

    from polygram import SAEFeatureRecord
    from polygram.confirmation.decoder_geometry import DecoderGeometryConfirmer
    from polygram.compression import (
        BlockSpec,
        Compressor,
        CompressionReport,
    )
    from polygram.config import CompressionConfig
    from polygram.encoding import MPSRung1, Rung5

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        # ---- Stage 1: download + slice SAE ----
        print(f"[1/7] download + slice SAE ({SAE_REPO})")
        t0 = time.time()
        full_sae = Path(hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILE))
        sliced_path = td_path / "sae_sliced.safetensors"
        slice_sae_to_features(full_sae, sliced_path,
                              list(range(N_FEATURES)))
        print(f"      sliced to {N_FEATURES} features ({time.time()-t0:.1f}s)")

        # ---- Stage 2: SAE → records + ValidationReport (decoder geometry) ----
        print("[2/7] DecoderGeometryConfirmer (real but cheap)")
        t0 = time.time()
        sae_state = load_file(str(sliced_path))
        W_dec = sae_state["W_dec"].astype(np.float64)  # (n_features, d_model)
        d_model = W_dec.shape[1]
        n_features = W_dec.shape[0]

        # Build SAEFeatureRecord dict per polygram's contract
        records: dict[int, SAEFeatureRecord] = {}
        for fid in range(n_features):
            records[fid] = SAEFeatureRecord(
                feature_id=fid,
                name=f"feat_{fid}",
                projection=W_dec[fid].astype(np.float64),
            )
        feature_ids = list(range(n_features))
        confirmer = DecoderGeometryConfirmer(
            records=records,
            sae_checkpoint=sliced_path,
            feature_ids=feature_ids,
            threshold=0.5,
        )
        vr = confirmer.run()
        print(f"      {len(vr.confirmed)} confirmed pairs across "
              f"{len(vr.pairs)} candidate pairs ({time.time()-t0:.1f}s)")

        if len(vr.confirmed) == 0:
            print("      WARN: no confirmed pairs at threshold 0.5; "
                  "trying lower thresholds...")
            for thr in (0.4, 0.3, 0.2, 0.1):
                confirmer = DecoderGeometryConfirmer(
                    records=records, sae_checkpoint=sliced_path,
                    feature_ids=feature_ids, threshold=thr,
                )
                vr = confirmer.run()
                print(f"      threshold={thr}: {len(vr.confirmed)} pairs")
                if len(vr.confirmed) >= 4:
                    break
        assert len(vr.confirmed) > 0, "no confirmed pairs at any threshold"

        # ---- Stage 3: heuristic heaviness score ----
        print("[3/7] compute heaviness score")
        decoder_norms = np.linalg.norm(W_dec, axis=1)
        pair_count = np.zeros(n_features, dtype=int)
        for (i, j) in vr.confirmed:
            pair_count[i] += 1
            pair_count[j] += 1
        # Heaviness = decoder_norm² (the "load-bearing" proxy in
        # production SAEs — features with high decoder norm carry more
        # of the reconstruction). On a 64-feature slice with only a
        # handful of confirmed pairs, pair_count is too sparse to
        # carry signal, so we fall back to decoder norm alone.
        heaviness = decoder_norms ** 2
        top_k = np.argsort(heaviness)[-HEAVY_K:][::-1].tolist()
        non_top = [i for i in range(n_features) if i not in top_k]
        print(f"      top-{HEAVY_K} heavy fids (by decoder_norm²): "
              f"{sorted(top_k)}")
        print(f"      heaviness[top-{HEAVY_K}]: "
              f"{[f'{heaviness[fid]:.3f}' for fid in sorted(top_k)]}")
        print(f"      heaviness[rest]: "
              f"mean={heaviness[non_top].mean():.3f}, "
              f"max={heaviness[non_top].max():.3f}, "
              f"min={heaviness[non_top].min():.3f}")

        # ---- Stage 4: build the partition ----
        heavy = BlockSpec(
            block_id="heavy", encoding_class="Rung5",
            encoding_kwargs={"n_amp_qubits": 4},
            feature_ids=tuple(sorted(top_k)),
        )
        tail = BlockSpec(
            block_id="tail", encoding_class="MPSRung1",
            feature_ids=tuple(
                fid for fid in range(n_features) if fid not in top_k
            ),
        )
        partition = (heavy, tail)

        # ---- Stage 5: Compressor.apply A vs B ----
        print("[4/7] Compressor.apply: Run A (uniform Rung5)")
        t0 = time.time()
        cfg_uniform = CompressionConfig(strategy="zero")
        c_uniform = Compressor(
            sae_checkpoint=sliced_path,
            validation_report=vr,
            config=cfg_uniform,
            encoding=Rung5(n_amp_qubits=4),
        )
        out_uniform = td_path / "compressed_uniform.safetensors"
        result_u = c_uniform.run(output_checkpoint=out_uniform)
        wall_u = time.time() - t0
        print(f"      uniform: n_kept={result_u.report.n_features_kept}, "
              f"n_zeroed={result_u.report.n_features_zeroed}, "
              f"clusters={result_u.report.n_clusters}, "
              f"scale_ratio={result_u.report.scale_compression_ratio:.4f}, "
              f"rank_ratio={result_u.report.rank_ratio:.3f}, "
              f"wall={wall_u:.1f}s")

        print("[5/7] Compressor.apply: Run B (partitioned heavy/tail)")
        t0 = time.time()
        cfg_partition = CompressionConfig(
            strategy="zero", encoding_partition=partition,
        )
        c_partition = Compressor(
            sae_checkpoint=sliced_path,
            validation_report=vr,
            config=cfg_partition,
            encoding=MPSRung1(),  # the rebuilt dict uses the partition's per-block encoding
        )
        out_partition = td_path / "compressed_partition.safetensors"
        result_p = c_partition.run(output_checkpoint=out_partition)
        wall_p = time.time() - t0
        rr_p = (f"{result_p.report.rank_ratio:.3f}"
                if result_p.report.rank_ratio is not None else "N/A")
        print(f"      partitioned: n_kept={result_p.report.n_features_kept}, "
              f"n_zeroed={result_p.report.n_features_zeroed}, "
              f"clusters={result_p.report.n_clusters}, "
              f"scale_ratio={result_p.report.scale_compression_ratio:.4f}, "
              f"rank_ratio={rr_p}, "
              f"wall={wall_p:.1f}s")
        for b in result_p.report.blocks:
            slot = _slot_cost(b.encoding_class, b.encoding_kwargs)
            rr = f"{b.rank_ratio:.3f}" if b.rank_ratio is not None else "N/A"
            pa = f"{b.post_A:.4f}" if b.post_A is not None else "N/A"
            print(f"        {b.block_id}: encoding={b.encoding_class}, "
                  f"n_kept={b.n_features_kept}, n_clusters={b.n_clusters}, "
                  f"slot_cost={slot}, rank_ratio={rr}, post_A={pa}")

        # ---- Stage 6: substrate cost comparison ----
        print("[6/7] substrate cost comparison")
        # Uniform: every kept feature uses Rung5's 128-slot budget
        uniform_substrate_kept = 128 * result_u.report.n_features_kept
        uniform_substrate_full = 128 * n_features
        # Partitioned: per-block
        partition_substrate_kept = sum(
            _slot_cost(b.encoding_class, b.encoding_kwargs) * b.n_features_kept
            for b in result_p.report.blocks
        )
        partition_substrate_full = sum(
            _slot_cost(b.encoding_class, b.encoding_kwargs) * len(b.feature_ids)
            for b in result_p.report.blocks
        )
        print(f"      substrate (kept reps):   uniform={uniform_substrate_kept}  "
              f"partition={partition_substrate_kept}  "
              f"reduction={uniform_substrate_kept/max(1,partition_substrate_kept):.2f}x")
        print(f"      substrate (full SAE):    uniform={uniform_substrate_full}  "
              f"partition={partition_substrate_full}  "
              f"reduction={uniform_substrate_full/max(1,partition_substrate_full):.2f}x")

        # ---- Stage 7: forge comparison (if sae-forge available) ----
        print("[7/7] forge comparison (sae-forge ForgePipeline.run_synthetic)")
        try:
            from saeforge import FeatureBasis, ForgePipeline, SubspaceProjector
            t0 = time.time()
            basis_u = FeatureBasis.from_polygram_checkpoint(out_uniform)
            basis_p = FeatureBasis.from_polygram_checkpoint(out_partition)
            print(f"      basis_u: n_features={basis_u.n_features}, d_model={basis_u.d_model}")
            print(f"      basis_p: n_features={basis_p.n_features}, d_model={basis_p.d_model}")

            def _run_forge(basis, label):
                proj = SubspaceProjector(basis, scale_boost="auto")
                pipeline = ForgePipeline(
                    basis=basis, projector=proj,
                    host_model_id="gpt2",
                    eval_prompts=[
                        "The mitochondrion is the powerhouse of the",
                        "All happy families are alike; each unhappy family is",
                    ],
                    dtype="float32", device="cpu",
                )
                t = time.time()
                forge_dir = td_path / f"forge_{label}"
                result = pipeline.run(forge_dir)
                wall = time.time() - t
                return result, wall

            r_u, w_u = _run_forge(basis_u, "uniform")
            r_p, w_p = _run_forge(basis_p, "partition")
            print(f"      uniform forge:     KL={r_u.faithfulness:.4f}, "
                  f"n_params={r_u.n_params}, wall={w_u:.1f}s")
            print(f"      partitioned forge: KL={r_p.faithfulness:.4f}, "
                  f"n_params={r_p.n_params}, wall={w_p:.1f}s")
            delta_kl_pct = 100 * (r_p.faithfulness - r_u.faithfulness) / max(1e-9, r_u.faithfulness)
            print(f"      Δ faithfulness_KL: {delta_kl_pct:+.2f}% "
                  f"(partition - uniform; negative = partition wins)")
            forge_ok = True
        except Exception as e:
            print(f"      forge skipped: {e}")
            r_u = r_p = None
            forge_ok = False

        # ---- Persist results ----
        summary = {
            "sae_repo": SAE_REPO,
            "sae_file": SAE_FILE,
            "n_features": n_features,
            "d_model": d_model,
            "heavy_k": HEAVY_K,
            "confirmer": "DecoderGeometryConfirmer",
            "threshold": float(confirmer.threshold),
            "n_confirmed_pairs": len(vr.confirmed),
            "heaviness_score": "decoder_norm² × pair_count",
            "top_k_heavy_fids": sorted(top_k),
            "uniform": {
                "encoding": "Rung5(n_amp_qubits=4)",
                "n_features_kept": result_u.report.n_features_kept,
                "n_features_zeroed": result_u.report.n_features_zeroed,
                "n_clusters": result_u.report.n_clusters,
                "scale_compression_ratio": result_u.report.scale_compression_ratio,
                "rank_ratio": result_u.report.rank_ratio,
                "substrate_kept_slots": uniform_substrate_kept,
                "substrate_full_slots": uniform_substrate_full,
                "wall_s": wall_u,
                "forge_faithfulness_kl": (r_u.faithfulness if forge_ok else None),
                "forge_n_params": (r_u.n_params if forge_ok else None),
            },
            "partition": {
                "blocks": [
                    {
                        "block_id":      b.block_id,
                        "encoding":      f"{b.encoding_class}({b.encoding_kwargs})",
                        "slot_cost":     _slot_cost(b.encoding_class, b.encoding_kwargs),
                        "n_features":    len(b.feature_ids),
                        "n_kept":        b.n_features_kept,
                        "n_zeroed":      b.n_features_zeroed,
                        "n_clusters":    b.n_clusters,
                        "scale_compression_ratio": b.scale_compression_ratio,
                        "rank_ratio":    b.rank_ratio,
                        "post_A":        b.post_A,
                        "informative_metric": b.informative_metric,
                    }
                    for b in result_p.report.blocks
                ],
                "n_features_kept": result_p.report.n_features_kept,
                "n_features_zeroed": result_p.report.n_features_zeroed,
                "n_clusters": result_p.report.n_clusters,
                "scale_compression_ratio": result_p.report.scale_compression_ratio,
                "rank_ratio": result_p.report.rank_ratio,
                "substrate_kept_slots": partition_substrate_kept,
                "substrate_full_slots": partition_substrate_full,
                "wall_s": wall_p,
                "forge_faithfulness_kl": (r_p.faithfulness if forge_ok else None),
                "forge_n_params": (r_p.n_params if forge_ok else None),
            },
            "substrate_reductions": {
                "kept_reps": uniform_substrate_kept / max(1, partition_substrate_kept),
                "full_input_sae": uniform_substrate_full / max(1, partition_substrate_full),
            },
        }
        out = Path("/tmp/real_partition_experiment.json")
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
