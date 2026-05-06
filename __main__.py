"""
refacer.__main__
~~~~~~~~~~~~~~~~
CLI entrypoint.  Invoked via:

    python -m refacer --input /path/to/photos --output /path/to/output

Run `python -m refacer --help` for full usage.
"""

import argparse
import logging
import os
import sys


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="refacer",
        description=(
            "Refacer — batch face anonymization and metadata scrubbing.\n"
            "All processing is fully offline; no network calls are made at runtime."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        metavar="DIR",
        help="Directory containing source images to anonymise.",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="DIR",
        help="Directory to write anonymised images to (created if absent).",
    )
    parser.add_argument(
        "--models",
        metavar="DIR",
        default=None,
        help=(
            "Directory containing model weight files "
            "(default: models/ inside the refacer package)."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser


def _resolve_models_dir(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    models_dir = _resolve_models_dir(args.models)

    # --- Load models (exits with clear message on missing weights) ---
    from refacer.models import load_models

    try:
        models = load_models(models_dir)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        sys.exit(1)
    except ImportError as exc:
        print(f"\nERROR: Missing dependency — {exc}\n", file=sys.stderr)
        sys.exit(1)

    # --- Run pipeline ---
    from refacer import pipeline

    stats = pipeline.run(
        input_dir=args.input,
        output_dir=args.output,
        models=models,
    )

    print(stats)

    # Exit non-zero if every image failed
    if stats.total > 0 and stats.failed == stats.total:
        sys.exit(1)


if __name__ == "__main__":
    main()