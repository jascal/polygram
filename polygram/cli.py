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
import sys
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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
