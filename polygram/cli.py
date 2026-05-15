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

    from_sae_lens_kwargs: dict[str, Any] = {
        # Preserve the CLI's existing contract — "without --assign-gamma,
        # every feature gets γ=0" — even after polygram's library-level
        # default flipped to True. The CLI flag stays the user-facing
        # opt-in; passing it explicit-False here insulates the CLI from
        # the library default change.
        "assign_gamma": bool(args.assign_gamma),
    }
    if args.n_clusters is not None:
        from_sae_lens_kwargs["n_clusters"] = args.n_clusters

    try:
        prediction = predict_cancellation_depth(
            records, feature_ids, **from_sae_lens_kwargs
        )
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


def _positive_int_argtype(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        ) from None
    if n < 1:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer (>= 1), got {n}"
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

    feature_ids = (
        _parse_feature_ids(args.features) if args.features is not None else None
    )

    try:
        # When --features is supplied, propagate to the lazy-slice path
        # so we don't read the full decoder tensor just to filter it
        # down. For GB-class SAEs this is the difference between
        # OOM and a sub-MB read.
        records = load_sae_safetensors(
            src, names=names, feature_ids=feature_ids
        )
    except (ValueError, ImportError) as exc:
        raise SystemExit(f"polygram: sae-import failed: {exc}") from None

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


def _read_prompts_file(path: Path) -> list[str]:
    """Read a prompts file, stripping `#`-prefixed and empty lines."""
    out: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out


def _cmd_validate(args: argparse.Namespace) -> int:
    from polygram.behavioural import BehaviouralValidator
    from polygram.sae_import import from_sae_lens, load_toy_sae

    dict_path = Path(args.dictionary)
    if not dict_path.is_file():
        sys.stderr.write(
            f"polygram: --dictionary file not found: {dict_path}\n"
        )
        return 2

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        sys.stderr.write(
            f"polygram: --sae-checkpoint file not found: {sae_path}\n"
        )
        return 2

    prompts_path = Path(args.prompts)
    if not prompts_path.is_file():
        sys.stderr.write(
            f"polygram: --prompts file not found: {prompts_path}\n"
        )
        return 2

    feature_ids = _parse_feature_ids(args.feature_ids)
    prompts = _read_prompts_file(prompts_path)
    if not prompts:
        sys.stderr.write(
            f"polygram: --prompts file is empty (no non-comment, "
            f"non-blank lines): {prompts_path}\n"
        )
        return 2

    try:
        records = load_toy_sae(dict_path)
    except (ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"polygram: failed to load --dictionary: {exc}\n")
        return 2

    n_dict_features = len(records)
    if n_dict_features != len(feature_ids):
        sys.stderr.write(
            f"polygram: --feature-ids supplied {len(feature_ids)} ids, "
            f"but --dictionary {dict_path} declares "
            f"{n_dict_features} features. Lengths must match.\n"
        )
        return 2

    try:
        dictionary, _selection_report = from_sae_lens(
            records, feature_ids, assign_gamma=True
        )
    except ValueError as exc:
        sys.stderr.write(f"polygram: from_sae_lens failed: {exc}\n")
        return 2

    try:
        validator = BehaviouralValidator(
            dictionary=dictionary,
            sae_checkpoint=sae_path,
            feature_ids=feature_ids,
            prompts=prompts,
            layer=args.layer,
            model_name=args.model,
            polygram_overlap_threshold=args.polygram_threshold,
            jaccard_threshold=args.jaccard_threshold,
            min_firing_rate=args.min_firing_rate,
            min_both_fire=args.min_both_fire,
            allow_layer_zero=args.allow_layer_zero,
            device=args.device,
        )
    except ValueError as exc:
        sys.stderr.write(f"polygram validate: {exc}\n")
        return 2

    sys.stderr.write("polygram validate: predict ...\n")
    sys.stderr.write(
        f"polygram validate: load model {args.model!r} ...\n"
    )
    sys.stderr.write(
        f"polygram validate: forward {len(prompts)} prompts ...\n"
    )
    sys.stderr.write("polygram validate: SAE encode ...\n")
    sys.stderr.write(
        f"polygram validate: ablation 1/{len(feature_ids)} ... "
        f"{len(feature_ids)}/{len(feature_ids)} ...\n"
    )
    sys.stderr.write("polygram validate: aggregate ...\n")

    try:
        report = validator.run()
    except ImportError as exc:
        sys.stderr.write(f"polygram validate: {exc}\n")
        return 2

    out_json = Path(args.output).resolve()
    report.to_json(out_json)
    sys.stderr.write(f"polygram validate: wrote JSON → {out_json}\n")

    if args.csv is not None:
        out_csv = Path(args.csv).resolve()
        report.to_csv(out_csv)
        sys.stderr.write(f"polygram validate: wrote CSV  → {out_csv}\n")

    return 0


def _parse_representatives(raw: str | None) -> dict[int, int] | None:
    """Parse `--representatives 0=12999,1=4192` into {0: 12999, 1: 4192}.

    Returns None on None input. Raises SystemExit(2) on malformed input.
    """
    if raw is None:
        return None
    out: dict[int, int] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            sys.stderr.write(
                f"polygram compress: --representatives entries must be "
                f"`cluster_id=fid`; got {part!r}\n"
            )
            raise SystemExit(2)
        cid_str, fid_str = part.split("=", 1)
        try:
            cid = int(cid_str)
            fid = int(fid_str)
        except ValueError:
            sys.stderr.write(
                f"polygram compress: --representatives entry {part!r} "
                f"has non-integer cluster_id or fid\n"
            )
            raise SystemExit(2) from None
        if cid in out:
            sys.stderr.write(
                f"polygram compress: --representatives cluster_id {cid} "
                f"specified more than once\n"
            )
            raise SystemExit(2)
        out[cid] = fid
    return out


def _cmd_compress(args: argparse.Namespace) -> int:
    from polygram.behavioural import ValidationReport
    from polygram.compression import Compressor
    from polygram.config import CompressionConfig

    vreport_path = Path(args.validation_report)
    if not vreport_path.is_file():
        sys.stderr.write(
            f"polygram compress: --validation-report file not found: "
            f"{vreport_path}\n"
        )
        return 2

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        sys.stderr.write(
            f"polygram compress: --sae-checkpoint file not found: "
            f"{sae_path}\n"
        )
        return 2

    # Pareto mode uses --output as a directory; threshold and
    # target-features modes use it as a JSON file path. Validate each
    # mode's required flags upfront so we don't half-do work.
    if args.pareto is not None:
        return _cmd_compress_pareto(args, sae_path, vreport_path)

    # Threshold or --target-features: existing flag shape.
    if not args.output_checkpoint:
        sys.stderr.write(
            "polygram compress: --output-checkpoint is required for "
            "threshold and --target-features modes\n"
        )
        return 2
    out_ckpt = Path(args.output_checkpoint).resolve()
    if out_ckpt == sae_path.resolve():
        sys.stderr.write(
            f"polygram compress: --output-checkpoint must differ from "
            f"--sae-checkpoint (both resolved to {out_ckpt})\n"
        )
        return 2

    try:
        representatives = _parse_representatives(args.representatives)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    sys.stderr.write("polygram compress: load validation report ...\n")
    try:
        vreport = ValidationReport.from_json(vreport_path)
    except (ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(
            f"polygram compress: failed to parse --validation-report: "
            f"{exc}\n"
        )
        return 2

    target_k = args.target_features
    if target_k is not None and target_k < 1:
        sys.stderr.write(
            f"polygram compress: --target-features must be a positive "
            f"integer; got {target_k!r}\n"
        )
        return 2

    sys.stderr.write("polygram compress: build plan ...\n")
    try:
        config = (
            CompressionConfig(
                target_n_features_kept=target_k,
                score_field=args.score_field,
            )
            if target_k is not None
            else None
        )
        compressor = Compressor(
            validation_report=vreport,
            sae_checkpoint=sae_path,
            strategy=args.strategy,
            representatives=representatives,
            config=config,
        )
    except ValueError as exc:
        sys.stderr.write(f"polygram compress: {exc}\n")
        return 2

    sys.stderr.write(
        f"polygram compress: rewrite weights → {out_ckpt} ...\n"
    )
    try:
        if target_k is not None:
            plan = compressor.plan_with_target()
            result = compressor.apply(
                plan=plan, output_checkpoint=out_ckpt
            )
        else:
            result = compressor.run(out_ckpt)
    except (ValueError, KeyError, IndexError) as exc:
        sys.stderr.write(f"polygram compress: {exc}\n")
        return 2

    out_json = Path(args.output).resolve()
    result.report.to_json(out_json)
    sys.stderr.write(
        f"polygram compress: wrote report → {out_json}\n"
    )
    sys.stderr.write(
        f"polygram compress: source sha256={result.report.source_checkpoint_sha256[:12]}… "
        f"output sha256={result.report.output_checkpoint_sha256[:12]}… "
        f"clusters={result.report.n_clusters} "
        f"zeroed={result.report.n_features_zeroed} "
        f"kept={result.report.n_features_kept}"
    )
    if target_k is not None:
        sys.stderr.write(
            f" target_k={target_k} "
            f"reached={'yes' if result.report.n_features_kept <= target_k else 'no'}"
        )
    sys.stderr.write("\n")
    return 0


def _cmd_compress_pareto(
    args: argparse.Namespace, sae_path: Path, vreport_path: Path
) -> int:
    from polygram.behavioural import ValidationReport
    from polygram.compression import Compressor
    from polygram.config import CompressionConfig

    try:
        targets = _parse_pareto_targets(args.pareto)
    except ValueError as exc:
        sys.stderr.write(f"polygram compress: {exc}\n")
        return 2

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.is_file():
        sys.stderr.write(
            f"polygram compress: --output must be a directory in "
            f"--pareto mode; got file: {out_dir}\n"
        )
        return 2

    try:
        representatives = _parse_representatives(args.representatives)
    except SystemExit as exc:
        return int(exc.code) if exc.code is not None else 2

    sys.stderr.write("polygram compress: load validation report ...\n")
    try:
        vreport = ValidationReport.from_json(vreport_path)
    except (ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(
            f"polygram compress: failed to parse --validation-report: "
            f"{exc}\n"
        )
        return 2

    try:
        compressor = Compressor(
            validation_report=vreport,
            sae_checkpoint=sae_path,
            strategy=args.strategy,
            representatives=representatives,
            config=CompressionConfig(score_field=args.score_field),
        )
    except ValueError as exc:
        sys.stderr.write(f"polygram compress: {exc}\n")
        return 2

    sys.stderr.write(
        f"polygram compress: plan_pareto(targets={targets}) ...\n"
    )
    try:
        pareto = compressor.plan_pareto(targets)
    except (ValueError, KeyError, IndexError) as exc:
        sys.stderr.write(f"polygram compress: {exc}\n")
        return 2

    pareto_json_path = out_dir / "pareto.json"
    pareto.to_json(pareto_json_path)
    sys.stderr.write(
        f"polygram compress: wrote pareto plan → {pareto_json_path}\n"
    )
    for outcome in pareto.outcomes:
        sys.stderr.write(
            f"polygram compress:   K={outcome.target_k} "
            f"kept={outcome.plan.n_features_kept} "
            f"reached={'yes' if outcome.reached_target else 'no'}\n"
        )

    if args.pareto_materialize:
        per_k_dir = out_dir / "pareto"
        per_k_dir.mkdir(parents=True, exist_ok=True)
        for outcome in pareto.outcomes:
            ckpt_path = per_k_dir / f"k_{outcome.target_k}.safetensors"
            if ckpt_path.resolve() == sae_path.resolve():
                sys.stderr.write(
                    f"polygram compress: materialise path collides with "
                    f"--sae-checkpoint at {ckpt_path}\n"
                )
                return 2
            sys.stderr.write(
                f"polygram compress: materialise K={outcome.target_k} "
                f"→ {ckpt_path} ...\n"
            )
            try:
                compressor.apply(
                    plan=outcome.plan, output_checkpoint=ckpt_path
                )
            except (ValueError, KeyError, IndexError) as exc:
                sys.stderr.write(f"polygram compress: {exc}\n")
                return 2

    return 0


def _parse_pareto_targets(raw: str) -> list[int]:
    """Parse a comma-separated list of positive integers for --pareto.

    Raises `ValueError` (caught by the handler) for empty, non-integer,
    or non-positive entries.
    """
    if not raw or not raw.strip():
        raise ValueError("--pareto: empty target list")
    parts = [s.strip() for s in raw.split(",") if s.strip()]
    if not parts:
        raise ValueError("--pareto: empty target list")
    targets: list[int] = []
    for s in parts:
        try:
            k = int(s)
        except ValueError as exc:
            raise ValueError(
                f"--pareto: cannot parse {s!r} as integer ({exc})"
            ) from exc
        if k < 1:
            raise ValueError(
                f"--pareto: every K must be a positive integer; got {k}"
            )
        targets.append(k)
    return targets


def _cmd_regrow(args: argparse.Namespace) -> int:
    from polygram.compression import CompressionReport, Regrower

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        sys.stderr.write(
            f"polygram regrow: --sae-checkpoint file not found: {sae_path}\n"
        )
        return 2

    out_ckpt = Path(args.output_checkpoint).resolve()
    if out_ckpt == sae_path.resolve():
        sys.stderr.write(
            f"polygram regrow: --output-checkpoint must differ from "
            f"--sae-checkpoint (both resolved to {out_ckpt})\n"
        )
        return 2

    # Mutually-exclusive groups: zeroed source
    if (args.zeroed_list is not None) == (args.compression_report is not None):
        sys.stderr.write(
            "polygram regrow: exactly one of --zeroed-list or "
            "--compression-report must be supplied\n"
        )
        return 2

    # Mutually-exclusive groups: residual source
    if (args.cached_residuals is not None) == (args.prompts is not None):
        sys.stderr.write(
            "polygram regrow: exactly one of --cached-residuals or "
            "--prompts must be supplied\n"
        )
        return 2

    # Resolve zeroed set
    zeroed: set[int] = set()
    upstream_report: CompressionReport | None = None
    if args.zeroed_list is not None:
        try:
            zeroed = set(_parse_feature_ids(args.zeroed_list))
        except SystemExit:
            sys.stderr.write(
                f"polygram regrow: malformed --zeroed-list: {args.zeroed_list!r}\n"
            )
            return 2
    else:
        report_path = Path(args.compression_report)
        if not report_path.is_file():
            sys.stderr.write(
                f"polygram regrow: --compression-report file not found: "
                f"{report_path}\n"
            )
            return 2
        try:
            upstream_report = CompressionReport.from_json(report_path)
        except (ValueError, json.JSONDecodeError) as exc:
            sys.stderr.write(
                f"polygram regrow: failed to parse --compression-report: "
                f"{exc}\n"
            )
            return 2

    # Resolve residual stream source
    cached_residuals = None
    prompts = None
    if args.cached_residuals is not None:
        residuals_path = Path(args.cached_residuals)
        if not residuals_path.is_file():
            sys.stderr.write(
                f"polygram regrow: --cached-residuals file not found: "
                f"{residuals_path}\n"
            )
            return 2
        try:
            import numpy as np
            cached_residuals = np.load(residuals_path)
        except Exception as exc:
            sys.stderr.write(
                f"polygram regrow: failed to load --cached-residuals: {exc}\n"
            )
            return 2
        if cached_residuals.ndim != 2:
            sys.stderr.write(
                f"polygram regrow: --cached-residuals must be 2D "
                f"(n_tokens × d_model); got shape {cached_residuals.shape!r}\n"
            )
            return 2
        if cached_residuals.dtype.kind != "f":
            sys.stderr.write(
                f"polygram regrow: --cached-residuals dtype must be "
                f"float32 or float64; got {cached_residuals.dtype!r}\n"
            )
            return 2
    else:
        prompts_path = Path(args.prompts)
        if not prompts_path.is_file():
            sys.stderr.write(
                f"polygram regrow: --prompts file not found: {prompts_path}\n"
            )
            return 2
        prompts = _read_prompts_file(prompts_path)
        if not prompts:
            sys.stderr.write(
                f"polygram regrow: --prompts file is empty: {prompts_path}\n"
            )
            return 2

    sys.stderr.write("polygram regrow: building Regrower ...\n")
    try:
        if upstream_report is not None:
            regrower = Regrower.from_compression_report(
                upstream_report,
                sae_checkpoint=sae_path,
                strategy=args.strategy,
                cached_residuals=cached_residuals,
                prompts=prompts,
                seed=args.seed,
                n_init=args.n_init,
                model_name=args.model,
                layer=args.layer,
                device=args.device,
            )
        else:
            regrower = Regrower(
                sae_checkpoint=sae_path,
                strategy=args.strategy,
                zeroed=zeroed,
                cached_residuals=cached_residuals,
                prompts=prompts,
                seed=args.seed,
                n_init=args.n_init,
                model_name=args.model,
                layer=args.layer,
                device=args.device,
            )
    except ValueError as exc:
        sys.stderr.write(f"polygram regrow: {exc}\n")
        return 2

    sys.stderr.write("polygram regrow: planning (k-means on residuals) ...\n")
    sys.stderr.write(
        f"polygram regrow: rewriting weights → {out_ckpt} ...\n"
    )
    try:
        result = regrower.run(out_ckpt)
    except (ValueError, KeyError, IndexError, RuntimeError, NotImplementedError) as exc:
        sys.stderr.write(f"polygram regrow: {exc}\n")
        return 2

    out_json = Path(args.output).resolve()
    result.report.to_json(out_json)
    sys.stderr.write(f"polygram regrow: wrote report → {out_json}\n")
    sys.stderr.write(
        f"polygram regrow: source sha256={result.report.source_checkpoint_sha256[:12]}… "
        f"output sha256={result.report.output_checkpoint_sha256[:12]}… "
        f"n_slots_repopulated={result.report.n_slots_repopulated} "
        f"n_slots_left_zero={result.report.n_slots_left_zero}\n"
    )
    return 0


def _cmd_compress_epoch(args: argparse.Namespace) -> int:
    from polygram.compression import EpochCompressor

    sae_path = Path(args.sae_checkpoint)
    if not sae_path.is_file():
        sys.stderr.write(
            f"polygram compress-epoch: --sae-checkpoint not found: {sae_path}\n"
        )
        return 2

    out_ckpt = Path(args.output_checkpoint).resolve()
    if out_ckpt == sae_path.resolve():
        sys.stderr.write(
            f"polygram compress-epoch: --output-checkpoint must differ "
            f"from --sae-checkpoint (both resolved to {out_ckpt})\n"
        )
        return 2

    prompts_path = Path(args.prompts)
    if not prompts_path.is_file():
        sys.stderr.write(
            f"polygram compress-epoch: --prompts not found: {prompts_path}\n"
        )
        return 2
    prompts = _read_prompts_file(prompts_path)
    if not prompts:
        sys.stderr.write(
            f"polygram compress-epoch: --prompts file is empty: {prompts_path}\n"
        )
        return 2

    sys.stderr.write("polygram compress-epoch: building EpochCompressor ...\n")
    try:
        epoch = EpochCompressor(
            sae_checkpoint=sae_path,
            prompts=prompts,
            layer=args.layer,
            model_name=args.model,
            strategy=args.strategy,
            device=args.device,
            coverage_target=args.coverage_target,
            cosine_threshold=args.cosine_threshold,
            n_visits_per_feature=args.n_visits_per_feature,
            n_panels_max=args.n_panels_max,
            min_firing_rate=args.min_firing_rate,
            max_iterations=args.max_iterations,
            quality_delta_multiplier=args.quality_delta_multiplier,
            polygram_overlap_threshold=args.polygram_threshold,
            jaccard_threshold=args.jaccard_threshold,
            min_both_fire=args.min_both_fire,
            save_intermediate_reports=args.save_intermediate_reports,
            allow_layer_zero=args.allow_layer_zero,
        )
    except ValueError as exc:
        sys.stderr.write(f"polygram compress-epoch: {exc}\n")
        return 2

    sys.stderr.write(
        f"polygram compress-epoch: pre-pass (firing rates + residuals) "
        f"on {len(prompts)} prompts ...\n"
    )
    sys.stderr.write(
        f"polygram compress-epoch: iterating up to {args.max_iterations} "
        f"× ≤{args.n_panels_max} panels ...\n"
    )
    try:
        result = epoch.run(out_ckpt)
    except (ValueError, KeyError, IndexError, RuntimeError) as exc:
        sys.stderr.write(f"polygram compress-epoch: {exc}\n")
        return 2

    out_json = Path(args.output).resolve()
    result.report.to_json(out_json)
    sys.stderr.write(
        f"polygram compress-epoch: wrote report → {out_json}\n"
    )
    sys.stderr.write(
        f"polygram compress-epoch: source sha256={result.report.source_checkpoint_sha256[:12]}… "
        f"output sha256={result.report.output_checkpoint_sha256[:12]}… "
        f"convergence_reason={result.report.convergence_reason} "
        f"iterations={len(result.report.iterations)} "
        f"n_features_zeroed_total={result.report.n_features_zeroed_total} "
        f"coverage_achieved={result.report.coverage_achieved:.3f}\n"
    )
    return 0


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
    p_an.add_argument(
        "--assign-gamma", action="store_true",
        help="forward assign_gamma=True to from_sae_lens; per-cluster "
             "PCA on projection vectors derives each feature's γ. "
             "Without this flag every feature gets γ=0, which collapses "
             "within-cluster overlaps to 1.0 on diverse-projection "
             "inputs. Real-SAE workloads almost universally need it.",
    )
    p_an.add_argument(
        "--n-clusters", type=_positive_int_argtype, default=None,
        help="forward n_clusters=N to from_sae_lens (used when k-means "
             "is the cluster fallback). default delegates to "
             "from_sae_lens (currently 2). Must be >= 1.",
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

    p_val = sub.add_parser(
        "validate",
        help="run the BehaviouralValidator end-to-end and write a "
             "ValidationReport (JSON + optional CSV)",
    )
    p_val.add_argument(
        "--dictionary", required=True,
        help="path to a toy-SAE-schema JSON file (loaded via "
             "load_toy_sae); the Dictionary is built via from_sae_lens",
    )
    p_val.add_argument(
        "--sae-checkpoint", required=True,
        help="path to a .safetensors file with W_enc / b_enc / "
             "W_dec / b_dec",
    )
    p_val.add_argument(
        "--feature-ids", required=True,
        help="comma-separated SAE feature ids in the same order as "
             "the dictionary's features",
    )
    p_val.add_argument(
        "--prompts", required=True,
        help="path to a text file (one prompt per non-empty line; "
             "lines starting with `#` are ignored)",
    )
    p_val.add_argument(
        "--layer", required=True, type=int,
        help="transformer block index whose forward_pre hook the "
             "validator registers",
    )
    p_val.add_argument(
        "--model", default="gpt2",
        help="HF model name (default: gpt2; threshold defaults were "
             "calibrated on GPT-2 small — consider re-calibrating for "
             "other model families)",
    )
    p_val.add_argument(
        "--polygram-threshold", type=float, default=0.7,
        help="Polygram squared-overlap gate threshold (default: 0.7)",
    )
    p_val.add_argument(
        "--jaccard-threshold", type=float, default=0.30,
        help="co-firing Jaccard gate threshold (default: 0.30)",
    )
    p_val.add_argument(
        "--min-firing-rate", type=float, default=0.01,
        help="firing-rate floor for selection-warning (default: 0.01)",
    )
    p_val.add_argument(
        "--min-both-fire", type=int, default=5,
        help="both-fire token count needed for paired-KL "
             "definability (default: 5)",
    )
    p_val.add_argument(
        "--allow-layer-zero", action="store_true",
        help="permit layer == 0 with a runtime warning (the default "
             "rejects layer 0 per docs/research/deeper-layer-"
             "ablation-probe.md)",
    )
    p_val.add_argument(
        "--device", default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="torch device for the model + activations "
             "(default: auto = cuda → mps → cpu, with a CPU-fallback "
             "warning); explicit cuda/mps raises if the backend isn't "
             "available",
    )
    p_val.add_argument(
        "--output", required=True,
        help="JSON output path; ValidationReport.to_json(...)",
    )
    p_val.add_argument(
        "--csv", default=None,
        help="optional CSV output path; ValidationReport.to_csv(...)",
    )
    p_val.set_defaults(func=_cmd_validate)

    p_compress = sub.add_parser(
        "compress",
        help="apply the compression action to an SAE checkpoint, "
             "consuming a ValidationReport and emitting a "
             "CompressionReport plus a rewritten safetensors file. "
             "Threshold mode (default) consumes `confirmed` pairs and "
             "produces byte-identical output to pre-0.4 releases. "
             "Pass --target-features N to compress to ~N cluster "
             "representatives, or --pareto K1,K2,... to plan a Pareto "
             "sweep (cheap: writes only pareto.json by default — pass "
             "--pareto-materialize to also write one safetensors per K).",
    )
    p_compress.add_argument(
        "--validation-report", required=True,
        help="path to a ValidationReport JSON (the loop's upstream "
             "half emits one)",
    )
    p_compress.add_argument(
        "--sae-checkpoint", required=True,
        help="path to the source .safetensors with W_enc / b_enc / "
             "W_dec / b_dec",
    )
    p_compress.add_argument(
        "--output-checkpoint", default=None,
        help="path for the rewritten .safetensors. Required in "
             "threshold and --target-features modes; not used with "
             "--pareto (use --output as a directory in that case)",
    )
    p_compress.add_argument(
        "--strategy", default="zero", choices=("zero",),
        help="compression strategy (initial release: zero only; "
             "merge is deferred to a follow-up change). Defaults to "
             "'zero'.",
    )
    p_compress.add_argument(
        "--output", required=True,
        help="In threshold and --target-features modes, JSON output "
             "path for the CompressionReport. In --pareto mode, "
             "directory path where pareto.json is written (plus "
             "pareto/k_<K>.safetensors files when --pareto-materialize "
             "is also passed).",
    )
    p_compress.add_argument(
        "--representatives", default=None,
        help="optional comma-separated `cluster_id=fid` pairs that "
             "override the default representative pick "
             "(e.g. --representatives 0=12999,1=4192)",
    )
    # ---- Phase 3 of add-pareto-target-compression ---------------------
    mode_group = p_compress.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--target-features", type=int, default=None, metavar="N",
        help="target-K mode: compress to ~N cluster representatives "
             "via Compressor.plan_with_target(). Mutually exclusive "
             "with --pareto. The threshold path is bypassed; see "
             "openspec/changes/add-pareto-target-compression.",
    )
    mode_group.add_argument(
        "--pareto", default=None, metavar="K1,K2,K3,...",
        help="Pareto-sweep mode: comma-separated list of positive "
             "integer K values. Writes <output>/pareto.json describing "
             "one ParetoOutcome per K. By default, no SAE checkpoints "
             "are materialised — pass --pareto-materialize to write "
             "<output>/pareto/k_<K>.safetensors per K.",
    )
    p_compress.add_argument(
        "--pareto-materialize", action="store_true",
        help="With --pareto, also write one materialised SAE per K "
             "under <output>/pareto/. Ignored without --pareto. "
             "Opt-in because materialisation can be many GB on "
             "production-size SAEs.",
    )
    p_compress.add_argument(
        "--score-field", default="polygram_overlap",
        choices=("polygram_overlap", "jaccard", "decoder_overlap"),
        help="CandidatePair field used to sort the pair list in "
             "--target-features and --pareto modes. Only the three "
             "bounded [0, 1] similarity-like fields are accepted; "
             "KL- and count-based fields are excluded by Decision 3 "
             "of add-pareto-target-compression. Ignored in threshold "
             "mode.",
    )
    p_compress.set_defaults(func=_cmd_compress)

    p_regrow = sub.add_parser(
        "regrow",
        help="repopulate zeroed slots in a compressed SAE checkpoint "
             "with new directions extracted from the SAE's activation "
             "residuals (residual_kmeans strategy)",
    )
    p_regrow.add_argument(
        "--sae-checkpoint", required=True,
        help="path to the source .safetensors (typically the output "
             "of a prior `polygram compress` run)",
    )
    p_regrow.add_argument(
        "--output-checkpoint", required=True,
        help="path for the rewritten .safetensors with zeroed slots "
             "populated; must differ from --sae-checkpoint",
    )
    p_regrow.add_argument(
        "--output", required=True,
        help="JSON output path for the RegrowReport",
    )
    p_regrow.add_argument(
        "--strategy", required=True,
        choices=("residual_kmeans", "high_decoder_norm_random",
                 "orthogonal_noise_scaled"),
        help="regrow strategy (initial release: residual_kmeans only; "
             "the other two are reserved enum members raising "
             "NotImplementedError)",
    )
    p_regrow.add_argument(
        "--zeroed-list", default=None,
        help="comma-separated zeroed feature ids (mutually exclusive "
             "with --compression-report)",
    )
    p_regrow.add_argument(
        "--compression-report", default=None,
        help="path to a CompressionReport JSON; the union of every "
             "cluster's `zeroed` list becomes the regrower's zeroed "
             "set (mutually exclusive with --zeroed-list)",
    )
    p_regrow.add_argument(
        "--cached-residuals", default=None,
        help="path to a .npy file with pre-captured residuals "
             "(2D, n_tokens × d_model; mutually exclusive with "
             "--prompts)",
    )
    p_regrow.add_argument(
        "--prompts", default=None,
        help="path to a prompts text file; one prompt per non-empty, "
             "non-`#`-prefixed line (mutually exclusive with "
             "--cached-residuals)",
    )
    p_regrow.add_argument(
        "--layer", type=int, default=10,
        help="transformer block index whose forward_pre hook captures "
             "the residual stream (default: 10)",
    )
    p_regrow.add_argument(
        "--model", default="gpt2",
        help="HF model name for the residual capture path "
             "(default: gpt2; supports GPT-2, Llama, Gemma, and any "
             "AutoModelForCausalLM-compatible model)",
    )
    p_regrow.add_argument(
        "--device", default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="torch device for the residual capture (default: auto)",
    )
    p_regrow.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for k-means (default: 0)",
    )
    p_regrow.add_argument(
        "--n-init", type=int, default=4,
        help="sklearn KMeans n_init parameter (default: 4)",
    )
    p_regrow.set_defaults(func=_cmd_regrow)

    p_epoch = sub.add_parser(
        "compress-epoch",
        help="multi-panel compression orchestrator: scales the "
             "validate→compress loop across many panels with stable-"
             "cluster fixed-point iteration",
    )
    p_epoch.add_argument(
        "--sae-checkpoint", required=True,
        help="path to the source .safetensors",
    )
    p_epoch.add_argument(
        "--prompts", required=True,
        help="path to a prompts text file; one prompt per non-empty, "
             "non-`#`-prefixed line",
    )
    p_epoch.add_argument(
        "--output-checkpoint", required=True,
        help="path for the rewritten .safetensors; must differ from "
             "--sae-checkpoint",
    )
    p_epoch.add_argument(
        "--output", required=True,
        help="JSON output path for the EpochReport",
    )
    p_epoch.add_argument(
        "--layer", type=int, default=10,
        help="transformer block index whose forward_pre hook captures "
             "residuals (default: 10)",
    )
    p_epoch.add_argument(
        "--model", default="gpt2",
        help="HF model name (default: gpt2; supports GPT-2, Llama, "
             "Gemma, and any AutoModelForCausalLM-compatible model)",
    )
    p_epoch.add_argument(
        "--strategy", default="zero", choices=("zero",),
        help="compression strategy passed through to Compressor "
             "(default: zero)",
    )
    p_epoch.add_argument(
        "--device", default="auto",
        choices=("auto", "cuda", "mps", "cpu"),
        help="torch device (default: auto)",
    )
    p_epoch.add_argument(
        "--coverage-target", type=float, default=0.95,
        help="target fraction of cosine-similar pairs to cover "
             "(default: 0.95)",
    )
    p_epoch.add_argument(
        "--cosine-threshold", type=float, default=0.30,
        help="decoder-cosine threshold for the pair-coverage graph "
             "(default: 0.30)",
    )
    p_epoch.add_argument(
        "--n-visits-per-feature", type=int, default=3,
        help="cap on how many panels each feature can appear in "
             "(default: 3)",
    )
    p_epoch.add_argument(
        "--n-panels-max", type=int, default=1000,
        help="hard cap on total panels (default: 1000)",
    )
    p_epoch.add_argument(
        "--min-firing-rate", type=float, default=0.01,
        help="firing-rate floor for eligible features (default: 0.01)",
    )
    p_epoch.add_argument(
        "--max-iterations", type=int, default=5,
        help="hard cap on iteration count (default: 5)",
    )
    p_epoch.add_argument(
        "--quality-delta-multiplier", type=float, default=2.0,
        help="iteration k aborts if its cross-entropy delta exceeds "
             "this multiplier × the first iteration's delta "
             "(default: 2.0)",
    )
    p_epoch.add_argument(
        "--polygram-threshold", type=float, default=0.7,
        help="Polygram squared-overlap gate threshold (default: 0.7)",
    )
    p_epoch.add_argument(
        "--jaccard-threshold", type=float, default=0.30,
        help="co-firing Jaccard gate threshold (default: 0.30)",
    )
    p_epoch.add_argument(
        "--min-both-fire", type=int, default=5,
        help="both-fire token count required for paired-KL "
             "definability (default: 5)",
    )
    p_epoch.add_argument(
        "--save-intermediate-reports", action="store_true",
        help="persist per-panel ValidationReport JSONs alongside the "
             "EpochReport (default: off)",
    )
    p_epoch.add_argument(
        "--allow-layer-zero", action="store_true",
        help="permit layer == 0 (default: rejects per "
             "docs/research/deeper-layer-ablation-probe.md)",
    )
    p_epoch.set_defaults(func=_cmd_compress_epoch)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
