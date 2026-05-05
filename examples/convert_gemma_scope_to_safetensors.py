"""Convert a Gemma-Scope ``params.npz`` SAE checkpoint into the
``W_dec``-keyed safetensors layout that ``load_sae_safetensors``
already understands.

Gemma-Scope (``google/gemma-scope-*``) ships every SAE as a NumPy zip
with lowercase keys ``w_enc``, ``w_dec``, ``b_enc``, ``b_dec``,
``threshold`` (JumpReLU). For projection-geometry work we only need
``w_dec`` — its rows are unit-norm feature directions in d_model
space, matching the convention the SAELens GPT-2 SAEs use after the
loader's ``W_dec`` rename.

Usage::

    huggingface-cli download google/gemma-scope-2b-pt-res \\
      layer_12/width_16k/average_l0_72/params.npz \\
      --local-dir scratch/gemma-scope/

    python examples/convert_gemma_scope_to_safetensors.py \\
      --input scratch/gemma-scope/layer_12/width_16k/average_l0_72/params.npz \\
      --output scratch/gemma-scope/layer_12_w16k_l0_72.safetensors

The output file is then a drop-in for any pipeline that calls
``load_sae_safetensors`` — including
``cross_encoding_stability_truly_insane.py`` via ``--sae-path``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


_DECODER_KEY_CANDIDATES = ("w_dec", "W_dec", "decoder", "dec")


def convert(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Read ``params.npz`` at ``input_path`` and write a safetensors
    file at ``output_path`` containing a single ``W_dec`` tensor.

    Returns ``(d_sae, d_model)`` for caller logging."""
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

    if w_dec.ndim != 2:
        raise ValueError(
            f"decoder tensor at key {matched!r} has shape {w_dec.shape}; "
            f"expected 2D (d_sae, d_model)"
        )

    # Gemma-Scope ships (d_sae, d_model) — same convention the loader
    # treats as W_dec. Heuristic guard: if rows < cols, that's almost
    # certainly already (d_sae, d_model); if rows > cols, also likely
    # (d_sae, d_model) since SAEs are over-complete. So no transpose
    # path here — but warn loudly if the shape looks suspicious.
    d_sae, d_model = w_dec.shape
    if d_sae < d_model:
        print(
            f"warning: shape {w_dec.shape} has d_sae < d_model. "
            f"Gemma-Scope SAEs are typically over-complete (d_sae > d_model). "
            f"Verify this is the intended checkpoint.",
            file=sys.stderr,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"W_dec": w_dec.astype(np.float32, copy=False)},
        str(output_path),
    )
    return d_sae, d_model


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
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"error: input not found: {args.input}", file=sys.stderr)
        return 2

    d_sae, d_model = convert(args.input, args.output)
    print(
        f"wrote {args.output} — W_dec shape ({d_sae}, {d_model})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
