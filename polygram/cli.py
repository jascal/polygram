"""polygram CLI — `polygram run <target>` and `polygram --version`.

Targets resolve as either:
- a filesystem path to a .py file (loaded via importlib.util), or
- a `pkg.module:callable` reference (loaded via importlib.import_module).

The loaded module / object MUST expose a `main(output_dir=...)` callable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from polygram import __version__


def _load_target(target: str) -> tuple[Any, str]:
    """Resolve a CLI target string to a (callable, source_label) pair.

    The returned callable accepts at least `output_dir`. Raises
    `SystemExit(2)` with a clear message on resolution failure.
    """
    if ":" in target and not Path(target).exists():
        module_name, _, attr = target.partition(":")
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise SystemExit(f"polygram: cannot import module {module_name!r}: {exc}") from None
        if not hasattr(module, attr):
            raise SystemExit(
                f"polygram: module {module_name!r} has no attribute {attr!r}"
            )
        obj = getattr(module, attr)
        if not callable(obj):
            raise SystemExit(f"polygram: {target} is not callable")
        return obj, target

    path = Path(target)
    if not path.exists():
        raise SystemExit(f"polygram: target not found: {target}")
    if path.suffix != ".py":
        raise SystemExit(f"polygram: expected a .py file or pkg.mod:func, got {target}")
    return _load_path_main(path), str(path)


def _load_path_main(path: Path) -> Callable[..., Any]:
    spec = importlib.util.spec_from_file_location(
        f"polygram_target_{path.stem}", path
    )
    if spec is None or spec.loader is None:
        raise SystemExit(f"polygram: could not load module from {path}")
    module: ModuleType = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SystemExit(f"polygram: error executing {path}: {exc}") from None
    if not hasattr(module, "main"):
        raise SystemExit(
            f"polygram: {path} has no `main(output_dir=...)` function"
        )
    return module.main


def _parse_feature_ids(raw: str) -> list[int]:
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            raise SystemExit(
                f"polygram: --features expects comma-separated ints, got {part!r}"
            ) from None
    if not out:
        raise SystemExit("polygram: --features must list at least one id")
    return out


def _cmd_analyze(args: argparse.Namespace) -> int:
    from polygram.analysis import (
        build_separation_graph,
        build_sharing_graph,
        predict_cancellation_depth,
        render_report,
    )
    from polygram.sae_import import load_toy_sae

    sae_path = Path(args.sae_path)
    if not sae_path.exists():
        raise SystemExit(f"polygram: SAE file not found: {sae_path}")

    feature_ids = _parse_feature_ids(args.features)
    records = load_toy_sae(sae_path)

    try:
        prediction = predict_cancellation_depth(records, feature_ids)
    except ValueError as exc:
        raise SystemExit(f"polygram: analyze failed: {exc}") from None

    report = render_report(
        prediction, sae_path=str(sae_path), feature_ids=feature_ids
    )
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(
        f"polygram: analyzed {len(feature_ids)} features → {out} "
        f"(score={prediction.encoding_suitability_score:.4f})"
    )

    if args.sharing_graph is not None:
        try:
            sharing = build_sharing_graph(
                prediction, threshold=args.sharing_threshold
            )
        except ValueError as exc:
            raise SystemExit(
                f"polygram: build_sharing_graph failed: {exc}"
            ) from None
        sharing_path = Path(args.sharing_graph).resolve()
        sharing_path.parent.mkdir(parents=True, exist_ok=True)
        sharing_path.write_text(sharing.to_json())
        print(
            f"polygram: wrote sharing graph → {sharing_path} "
            f"(threshold={args.sharing_threshold})"
        )

    if args.separation_graph is not None:
        try:
            separation = build_separation_graph(
                prediction, threshold=args.separation_threshold
            )
        except ValueError as exc:
            raise SystemExit(
                f"polygram: build_separation_graph failed: {exc}"
            ) from None
        separation_path = Path(args.separation_graph).resolve()
        separation_path.parent.mkdir(parents=True, exist_ok=True)
        separation_path.write_text(separation.to_json())
        print(
            f"polygram: wrote separation graph → {separation_path} "
            f"(threshold={args.separation_threshold})"
        )

    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    target_fn, label = _load_target(args.target)
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    kwargs: dict[str, Any] = {"output_dir": out}
    if args.n_points is not None:
        kwargs["n_points"] = args.n_points

    try:
        target_fn(**kwargs)
    except TypeError:
        if "n_points" in kwargs:
            kwargs.pop("n_points")
            target_fn(**kwargs)
        else:
            raise

    print(f"polygram: ran {label} → {out}")
    return 0


def _resolve_dictionary_ref(ref: str) -> Any:
    """Resolve `--dictionary REF` to a `Dictionary` instance.

    Accepts a `module:callable` form whose callable returns a
    Dictionary. `.q.orca.md` paths are surfaced today as a clear
    `SystemExit` pointing at the supported form — the rung-1
    .q.orca.md surface does not carry cluster information, so the
    inverse round-trip is a follow-up research-track task (tracked in
    `openspec/changes/tech-debt-backlog/`).
    """
    if Path(ref).suffix == ".q.orca.md" or (
        ":" not in ref and Path(ref).exists() and ref.endswith(".q.orca.md")
    ):
        raise SystemExit(
            "polygram: .q.orca.md → Dictionary round-trip is not yet "
            "implemented (rung-1 machines do not carry cluster info on "
            "the wire). Pass a `module:callable` reference instead, e.g. "
            "`examples.animals_hea:build_dictionary`."
        )
    if ":" not in ref:
        raise SystemExit(
            f"polygram: --dictionary {ref!r} must be `module:callable` "
            "(e.g. `examples.animals_hea:build_dictionary`)."
        )
    module_name, _, attr = ref.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise SystemExit(
            f"polygram: cannot import dictionary module {module_name!r}: {exc}"
        ) from None
    if not hasattr(module, attr):
        raise SystemExit(
            f"polygram: dictionary module {module_name!r} has no attribute "
            f"{attr!r}"
        )
    obj = getattr(module, attr)
    if not callable(obj):
        raise SystemExit(
            f"polygram: --dictionary {ref!r} is not callable"
        )
    try:
        result = obj()
    except Exception as exc:
        raise SystemExit(
            f"polygram: dictionary callable {ref!r} raised: {exc}"
        ) from None

    from polygram.dictionary import Dictionary

    if not isinstance(result, Dictionary):
        raise SystemExit(
            f"polygram: dictionary callable {ref!r} returned "
            f"{type(result).__name__}, expected Dictionary"
        )
    return result


def _topk_argtype(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--top-k expects an integer, got {value!r}"
        ) from None
    from polygram.batch import TOP_K_MAX, TOP_K_MIN

    if n < TOP_K_MIN or n > TOP_K_MAX:
        raise argparse.ArgumentTypeError(
            f"--top-k must be in [{TOP_K_MIN}, {TOP_K_MAX}]; got {n}"
        )
    return n


def _cmd_batch(args: argparse.Namespace) -> int:
    from polygram.analysis.feature_graph import FeatureGraph
    from polygram.batch import BatchExperiment

    graph_path = Path(args.feature_graph)
    if not graph_path.exists():
        raise SystemExit(
            f"polygram: --feature-graph file not found: {graph_path}"
        )
    try:
        graph = FeatureGraph.from_json(graph_path.read_text())
    except (json.JSONDecodeError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"polygram: failed to parse --feature-graph {graph_path}: {exc}"
        ) from None

    dictionary = _resolve_dictionary_ref(args.dictionary)

    if args.output_dir is not None:
        out_dir = Path(args.output_dir).resolve()
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="polygram-batch-"))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        experiment = BatchExperiment(
            feature_graph=graph,
            dictionary=dictionary,
            top_k=args.top_k,
            knobs=args.knobs,
            output_dir=out_dir,
        )
    except ValueError as exc:
        raise SystemExit(f"polygram: batch failed: {exc}") from None

    experiment.run()
    results_path = out_dir / "batch_results.json"
    print(f"polygram: wrote batch results → {results_path}")
    return 0


def _resolve_names_file(path: str) -> dict[int, str]:
    """Load a `--names` JSON file, auto-detect its shape, and return
    a `dict[int, str]` (`{id: name}`) suitable for
    `load_sae_safetensors(names=...)`.

    String-valued maps are interpreted as `{id: name}`; int-valued
    maps as `{name: id}` and inverted. Mixed-type values exit
    non-zero with a clear error.
    """
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"polygram: --names file not found: {path}")
    try:
        raw = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"polygram: failed to parse --names {path}: {exc}"
        ) from None
    if not isinstance(raw, dict):
        raise SystemExit(
            f"polygram: --names {path} must contain a JSON object, "
            f"got {type(raw).__name__}"
        )
    if not raw:
        return {}

    value_types = {type(v) for v in raw.values()}
    if len(value_types) != 1:
        raise SystemExit(
            f"polygram: --names {path} mixes value types {value_types!r}; "
            "expected a uniform `{id: name}` (string values) or "
            "`{name: id}` (int values) map"
        )
    sole = next(iter(value_types))
    if sole is str:
        try:
            return {int(k): str(v) for k, v in raw.items()}
        except ValueError as exc:
            raise SystemExit(
                f"polygram: --names {path}: keys must parse as ints "
                f"under the {{id: name}} shape: {exc}"
            ) from None
    if sole is int:
        return {int(v): str(k) for k, v in raw.items()}
    raise SystemExit(
        f"polygram: --names {path} value type {sole.__name__} not "
        "supported; use string values for `{id: name}` or int values "
        "for `{name: id}`"
    )


def _cmd_sae_import(args: argparse.Namespace) -> int:
    from polygram.sae_import import load_sae_safetensors

    src = Path(args.path)
    if not src.exists():
        raise SystemExit(f"polygram: SAE file not found: {src}")

    names = _resolve_names_file(args.names) if args.names is not None else None

    try:
        records = load_sae_safetensors(src, names=names)
    except (ValueError, ImportError) as exc:
        raise SystemExit(f"polygram: sae-import failed: {exc}") from None

    if args.features is not None:
        feature_ids = _parse_feature_ids(args.features)
        missing = [fid for fid in feature_ids if fid not in records]
        if missing:
            raise SystemExit(
                f"polygram: --features id(s) not in source SAE: "
                f"{missing} (valid range: [0, {len(records)}))"
            )
        records = {fid: records[fid] for fid in feature_ids}

    payload = {
        "schema_version": 1,
        "description": (
            f"Polygram sae-import — {src.name}, "
            f"{len(records)} features"
        ),
        "features": [
            _record_to_json(rec) for rec in records.values()
        ],
    }
    text = json.dumps(payload, indent=2)
    if args.output is None:
        print(text)
    else:
        out = Path(args.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        sys.stderr.write(f"polygram: wrote {len(records)} features → {out}\n")
    return 0


def _record_to_json(rec: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "feature_id": int(rec.feature_id),
        "name": rec.name,
        "projection": [float(x) for x in rec.projection.tolist()],
    }
    if rec.label is not None:
        out["label"] = rec.label
    if rec.activation_mean is not None:
        out["activation_mean"] = float(rec.activation_mean)
    if rec.activation_std is not None:
        out["activation_std"] = float(rec.activation_std)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="polygram")
    parser.add_argument(
        "--version", action="version", version=f"polygram {__version__}"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run an example or experiment module")
    p_run.add_argument(
        "target",
        help="path to a .py file (must define `main(output_dir=...)`) "
             "or pkg.module:callable",
    )
    p_run.add_argument(
        "--output-dir", default="examples/output",
        help="where the target writes artifacts (default: examples/output)",
    )
    p_run.add_argument(
        "--n-points", type=int, default=None,
        help="forwarded to target main() if it accepts it",
    )
    p_run.set_defaults(func=_cmd_run)

    p_an = sub.add_parser(
        "analyze",
        help="triage an SAE feature subset (predict structural floors + "
             "cancellation gaps; no quantum simulation)",
    )
    p_an.add_argument(
        "sae_path",
        help="path to a toy-SAE JSON file (schema matches "
             "tests/fixtures/toy_sae.json)",
    )
    p_an.add_argument(
        "--features", required=True,
        help="comma-separated feature ids to triage (≤8)",
    )
    p_an.add_argument(
        "--output", default="analysis_report.md",
        help="markdown report output path (default: analysis_report.md)",
    )
    p_an.add_argument(
        "--sharing-graph", default=None,
        help="when set, write the sharing FeatureGraph JSON to this path",
    )
    p_an.add_argument(
        "--sharing-threshold", type=float, default=0.5,
        help="sharing-graph weight threshold (default: 0.5; "
             "ignored unless --sharing-graph is set)",
    )
    p_an.add_argument(
        "--separation-graph", default=None,
        help="when set, write the separation FeatureGraph JSON to this path",
    )
    p_an.add_argument(
        "--separation-threshold", type=float, default=0.2,
        help="separation-graph weight threshold (default: 0.2; "
             "ignored unless --separation-graph is set)",
    )
    p_an.set_defaults(func=_cmd_analyze)

    p_batch = sub.add_parser(
        "batch",
        help="run Cancellation on the top-K edges of a FeatureGraph",
    )
    p_batch.add_argument(
        "--feature-graph", required=True,
        help="path to a JSON FeatureGraph (output of "
             "`build_sharing_graph` / `build_separation_graph`)",
    )
    p_batch.add_argument(
        "--dictionary", required=True,
        help="`module:callable` returning a Dictionary "
             "(e.g. `examples.animals_hea:build_dictionary`)",
    )
    p_batch.add_argument(
        "--top-k", type=_topk_argtype, default=8,
        help="number of input-graph edges to run (default 8; cap 16)",
    )
    p_batch.add_argument(
        "--knobs", choices=("cluster_shared", "per_feature"),
        default="cluster_shared",
        help="knob path style passed to per-pair Cancellation "
             "(default: cluster_shared)",
    )
    p_batch.add_argument(
        "--output-dir", default=None,
        help="where per-pair artifacts and batch_results.json land "
             "(default: a freshly-created temp directory)",
    )
    p_batch.set_defaults(func=_cmd_batch)

    p_sae = sub.add_parser(
        "sae-import",
        help="convert a .safetensors SAE checkpoint to the toy-SAE JSON "
             "schema consumed by `polygram analyze`",
    )
    p_sae.add_argument(
        "path",
        help="path to a .safetensors file containing decoder weights "
             "(W_dec / decoder.weight / dec)",
    )
    p_sae.add_argument(
        "--features", default=None,
        help="optional comma-separated feature ids to keep "
             "(default: every feature loaded)",
    )
    p_sae.add_argument(
        "--names", default=None,
        help="optional JSON file mapping {id: name} (string values) "
             "or {name: id} (int values); auto-detected by value type",
    )
    p_sae.add_argument(
        "--output", default=None,
        help="output path for the toy-SAE JSON document "
             "(default: stdout)",
    )
    p_sae.set_defaults(func=_cmd_sae_import)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
