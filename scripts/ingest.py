"""Data ingestion CLI entry point.

Usage::

    python scripts/ingest.py --path docs/file.pdf --collection mydocs
    python scripts/ingest.py --path docs/ --collection mydocs --force

Builds an IngestionPipeline from config/settings.yaml and ingests a single
file or every supported file under a directory. Re-running on unchanged files
is skipped automatically (SHA256 early-exit) unless --force is given.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.settings import load_settings
from src.ingestion.pipeline import IngestionError, IngestionPipeline, IngestionResult
from src.observability.logger import get_logger

# Importing the loader package registers the built-in loaders with the
# LoaderFactory; the supported extensions are derived from that registry so the
# CLI stays in sync with whatever formats are registered.
import src.libs.loader  # noqa: F401
from src.libs.loader.loader_factory import registered_extensions

logger = get_logger()


def _supported_extensions() -> set[str]:
    """Supported source extensions, derived from the LoaderFactory registry."""
    return registered_extensions()


def _collect_files(path: Path) -> list[Path]:
    """Return the list of supported files for a file or directory path."""
    supported = _supported_extensions()
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(
            p for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in supported
        )
    return []


def _print_progress(stage: str, current: int, total: int) -> None:
    """Simple stderr progress reporter."""
    print(f"  [{stage}] {current}/{total}", file=sys.stderr)


def run_ingestion(
    path: str,
    collection: str,
    force: bool,
    config_path: str = "config/settings.yaml",
    show_progress: bool = True,
) -> list[IngestionResult]:
    """Run ingestion for a path (file or directory). Returns per-file results."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Path not found: {path}")

    files = _collect_files(source)
    if not files:
        logger.warning("No supported files found at: %s", path)
        return []

    settings = load_settings(config_path)
    pipeline = IngestionPipeline.from_settings(settings)

    results: list[IngestionResult] = []
    for f in files:
        logger.info("Ingesting %s -> collection '%s'", f, collection)
        try:
            result = pipeline.run(
                str(f),
                collection=collection,
                force=force,
                on_progress=_print_progress if show_progress else None,
            )
            results.append(result)
            if result.skipped:
                logger.info("  Skipped (unchanged): %s", f)
            else:
                logger.info(
                    "  Done: %d chunks, %d images", result.total_chunks, result.total_images
                )
        except IngestionError as exc:
            logger.error("  Failed: %s", exc)
            results.append(
                IngestionResult(
                    source_path=str(f), collection=collection,
                    trace_id="", error=str(exc),
                )
            )
    return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest documents into the Smart Knowledge Hub."
    )
    parser.add_argument(
        "--path", required=True, help="Path to a file or directory to ingest."
    )
    parser.add_argument(
        "--collection", default="default", help="Target collection name."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force re-ingestion even if the file is unchanged.",
    )
    parser.add_argument(
        "--config", default="config/settings.yaml",
        help="Path to the settings YAML file.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    args = _build_parser().parse_args(argv)

    try:
        results = run_ingestion(
            path=args.path,
            collection=args.collection,
            force=args.force,
            config_path=args.config,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not results:
        return 1

    ingested = sum(1 for r in results if not r.skipped and not r.error)
    skipped = sum(1 for r in results if r.skipped)
    failed = sum(1 for r in results if r.error)
    logger.info(
        "Summary: %d ingested, %d skipped, %d failed", ingested, skipped, failed
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
