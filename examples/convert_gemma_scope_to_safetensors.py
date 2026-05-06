"""Convert a Gemma-Scope ``params.npz`` SAE checkpoint to safetensors.

By default all keys are preserved (``W_dec``, ``W_enc``, ``b_dec``,
``b_enc``, ``threshold``), producing a checkpoint that is compatible with
both ``load_sae_safetensors`` and :class:`~polygram.Compressor`.

Pass ``--dec-only`` to write only the ``W_dec`` tensor, which is sufficient
for projection-geometry work (``load_sae_safetensors``, Gram analysis) but
cannot be used as a compressor checkpoint.

Gemma-Scope (``google/gemma-scope-*``) ships SAEs as NumPy zips with keys
``w_enc``, ``w_dec``, ``b_enc``, ``b_dec``, ``threshold`` (JumpReLU). The
``W_dec`` rows are unit-norm feature directions in ``d_model`` space,
matching the convention the SAELens GPT-2 SAEs use after the loader's
``W_dec`` rename.

Full-convert → compress workflow::

    hf download google/gemma-scope-2b-pt-res \\
      embedding/width_4k/average_l0_21/params.npz \\
      --local-dir scratch/gemma-scope/

    python examples/convert_gemma_scope_to_safetensors.py \\
      --input scratch/gemma-scope/embedding/width_4k/average_l0_21/params.npz \\
      --output scratch/gemma-scope/embedding_w4k_l0_21_full.safetensors

    # then load, confirm, and compress:
    #   records = load_sae_safetensors("...full.safetensors")
    #   confirmer = DecoderGeometryConfirmer(records, "...full.safetensors", ids)
    #   Compressor(confirmer.run(), "...full.safetensors").run(output_checkpoint=...)

Projection-geometry only (``--dec-only``)::

    python examples/convert_gemma_scope_to_safetensors.py \\
      --input scratch/gemma-scope/embedding/width_4k/average_l0_21/params.npz \\
      --output scratch/gemma-scope/embedding_w4k_l0_21.safetensors \\
      --dec-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


_DECODER_KEY_CANDIDATES = ("w_dec", "W_dec", "decoder", "dec")

# Keys to preserve in a full conversion (lower-case source → upper-case target).
_FULL_KEY_MAP = {
    "w_dec": "W_dec",
    "W_dec": "W_dec",
    "w_enc": "W_enc",
    "W_enc": "W_enc",
    "b_dec": "b_dec",
    "b_enc": "b_enc",
    "threshold": "threshold",
}


def convert(
    input_path: Path,
    output_path: Path,
    *,
    dec_only: bool = False,
) -> tuple[int, int, list[str]]:
    """Read ``params.npz`` at ``input_path`` and write a safetensors file.

    Returns ``(d_sae, d_model, keys_written)``."""
    from safetensors.numpy import save_file

    with np.load(input_path) as data:
        keys = list(data.keys())
        matched = next((k for k in _DECODER_KEY_CANDIDATES if k in keys), None)
        if matched is None:
            raise KeyError(
                f"no decoder key found in {input_path}. "
                f"keys present: {keys}. "
                f"expected one of: {_DECODER_KEY_CANDIDATES}"
            )
        w_dec = np.asarray(data[matched])

        if dec_only:
            payload = {"W_dec": w_dec.astype(np.float32, copy=False)}
        else:
            payload = {}
            for src_key, dst_key in _FULL_KEY_MAP.items():
                if src_key in keys:
                    payload[dst_key] = np.asarray(data[src_key]).astype(
                        np.float32, copy=False
                    )
            if "W_dec" not in payload:
                payload["W_dec"] = w_dec.astype(np.float32, copy=False)

    if w_dec.ndim != 2:
        raise ValueError(
            f"decoder tensor at key {matched!r} has shape {w_dec.shape}; "
            f"expected 2D (d_sae, d_model)"
        )

    d_sae, d_model = w_dec.shape
    if d_sae < d_model:
        print(
            f"warning: shape {w_dec.shape} has d_sae < d_model. "
            f"Gemma-Scope SAEs are typically over-complete (d_sae > d_model). "
            f"Verify this is the intended checkpoint.",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(payload, str(output_path))
    return d_sae, d_model, sorted(payload.keys())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path,
        help="path to a Gemma-Scope params.npz file",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="destination .safetensors path",
    )
    parser.add_argument(
        "--dec-only", action="store_true",
        help="write only W_dec (sufficient for geometry work; not for compression)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    d_sae, d_model, keys_written = convert(
        args.input, args.output, dec_only=args.dec_only
    )
    keys_str = ", ".join(keys_written)
    print(f"wrote {args.output} — W_dec shape ({d_sae}, {d_model})  keys: {keys_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
