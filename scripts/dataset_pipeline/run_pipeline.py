"""Pipeline orchestrator for the RepoGraph AI dataset pipeline.

Runs discovery, cloning, metadata collection, and metrics extraction in order
while preserving resumable collection state between stages.
"""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

try:
    from scripts.dataset_pipeline.collection_state import CollectionStateManager
    from scripts.dataset_pipeline.github_search import OUTPUT_CSV_PATH, discover_repositories
    from scripts.dataset_pipeline.metadata_collector import MetadataCollector, METADATA_CSV_PATH
    from scripts.dataset_pipeline.metrics_collector import MetricsCollector, METRICS_CSV_PATH, REPOSITORIES_DIR
    from scripts.dataset_pipeline.repository_collector import RepositoryCollector
    from scripts.dataset_pipeline.utils import setup_pipeline_logging
except ImportError:
    from collection_state import CollectionStateManager
    from github_search import OUTPUT_CSV_PATH, discover_repositories
    from metadata_collector import MetadataCollector, METADATA_CSV_PATH
    from metrics_collector import MetricsCollector, METRICS_CSV_PATH, REPOSITORIES_DIR
    from repository_collector import RepositoryCollector
    from utils import setup_pipeline_logging

logger = setup_pipeline_logging(__name__)


StageCallable = Callable[[], Optional[Dict[str, Any]]]


def print_banner(stage_name: str) -> None:
    """Prints a readable stage banner."""
    line = "=" * 56
    print(f"\n{line}")
    print(stage_name)
    print(line)


def require_file(path: Path, description: str) -> None:
    """Validates that a required file exists."""
    if not path.exists():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def require_directory(path: Path, description: str) -> None:
    """Validates that a required directory exists."""
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Required {description} not found: {path}")


def count_csv_rows(path: Path) -> int:
    """Returns the number of records in a CSV file, or zero when absent."""
    if not path.exists():
        return 0
    return len(pd.read_csv(path))


def summarize_stage(stage_name: str, elapsed_seconds: float, summary: Optional[Dict[str, Any]]) -> None:
    """Prints a normalized post-stage summary."""
    summary = summary or {}
    repositories_processed = (
        summary.get("repositories_processed")
        or summary.get("repositories_discovered")
        or summary.get("total")
        or summary.get("total_records")
        or 0
    )
    failures = summary.get("failed", 0)
    skipped = summary.get("already_processed") or summary.get("already_existed") or 0

    print(f"\n{stage_name} complete")
    print(f"Elapsed time: {elapsed_seconds:.2f}s")
    print(f"Repositories processed: {repositories_processed}")
    print(f"Failures: {failures}")
    print(f"Skipped repositories: {skipped}")


def run_stage(
    stage_name: str,
    stage_func: StageCallable,
    validate_func: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Runs one pipeline stage with validation, logging, timing, and failure handling."""
    print_banner(stage_name)
    logger.info("Starting stage: %s", stage_name)

    if validate_func:
        validate_func()

    started_at = time.perf_counter()
    summary = stage_func() or {}
    elapsed_seconds = time.perf_counter() - started_at

    logger.info("Completed stage: %s in %.2fs", stage_name, elapsed_seconds)
    summarize_stage(stage_name, elapsed_seconds, summary)
    return summary


def run_discovery_stage() -> Dict[str, Any]:
    """Runs GitHub discovery and initializes collection state from candidates."""
    discover_repositories()
    require_file(OUTPUT_CSV_PATH, "repository candidates CSV")

    manager = CollectionStateManager()
    manager.initialize_state(OUTPUT_CSV_PATH)
    row_count = count_csv_rows(OUTPUT_CSV_PATH)
    return {
        "repositories_processed": row_count,
        "failed": 0,
        "already_processed": 0,
    }


def run_repository_collector_stage() -> Dict[str, Any]:
    """Runs repository cloning."""
    return RepositoryCollector().run()


def run_metadata_stage() -> Dict[str, Any]:
    """Runs GitHub metadata collection."""
    return MetadataCollector().run()


def run_metrics_stage() -> Dict[str, Any]:
    """Runs static metrics extraction."""
    return MetricsCollector().run()


def final_report(total_elapsed_seconds: float) -> None:
    """Prints the final pipeline report from collection state and output files."""
    manager = CollectionStateManager()
    state_summary = manager.export_summary()

    repositories_discovered = state_summary.get("discovered_completed", count_csv_rows(OUTPUT_CSV_PATH))
    repositories_cloned = state_summary.get("cloned_completed", 0)
    metadata_collected = state_summary.get("metadata_collected_completed", count_csv_rows(METADATA_CSV_PATH))
    metrics_extracted = state_summary.get("metrics_extracted_completed", count_csv_rows(METRICS_CSV_PATH))
    failed = state_summary.get("failed", 0)

    print("\n========================")
    print("Dataset Pipeline Complete")
    print(f"Repositories Discovered: {repositories_discovered}")
    print(f"Repositories Cloned: {repositories_cloned}")
    print(f"Metadata Collected: {metadata_collected}")
    print(f"Metrics Extracted: {metrics_extracted}")
    print(f"Failed: {failed}")
    print(f"Elapsed Time: {total_elapsed_seconds:.2f}s")
    print("========================")


def run_pipeline() -> None:
    """Runs the full RepoGraph AI dataset pipeline."""
    pipeline_started_at = time.perf_counter()

    stages = [
        (
            "1. GitHub Discovery",
            run_discovery_stage,
            None,
        ),
        (
            "2. Repository Collector",
            run_repository_collector_stage,
            lambda: require_file(OUTPUT_CSV_PATH, "repository candidates CSV"),
        ),
        (
            "3. Metadata Collector",
            run_metadata_stage,
            lambda: require_file(OUTPUT_CSV_PATH, "repository candidates CSV"),
        ),
        (
            "4. Metrics Collector",
            run_metrics_stage,
            lambda: require_directory(REPOSITORIES_DIR, "cloned repositories directory"),
        ),
    ]

    try:
        for stage_name, stage_func, validate_func in stages:
            run_stage(stage_name, stage_func, validate_func)
    except Exception:
        logger.exception("Pipeline stopped because a stage failed.")
        print("\nPipeline stopped because a stage failed. Collection state has been preserved.")
        raise

    total_elapsed_seconds = time.perf_counter() - pipeline_started_at
    final_report(total_elapsed_seconds)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_pipeline()
