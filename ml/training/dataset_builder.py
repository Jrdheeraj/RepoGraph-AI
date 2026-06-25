"""Dataset Builder for RepoGraph AI.

This module generates a training dataset from repositories.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Set

import pandas as pd

from ml.metrics_extractor import MetricsExtractor
from ml.complexity_scorer import ComplexityScorer

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")
_OUTPUT_CSV = Path("data/datasets/repositories.csv")


class DatasetBuilder:
    """Builder to generate a pandas DataFrame dataset from repositories."""

    def __init__(self, repositories_dir: str = "data/repositories/") -> None:
        """Initialize DatasetBuilder.

        Args:
            repositories_dir: Directory containing cloned repositories.
        """
        self.repositories_dir = Path(repositories_dir)
        self._dataset: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, repo_name: str) -> Path:
        """Return the cache file path for a given repository name."""
        return _CACHE_DIR / f"{repo_name}.json"

    def _load_cache(self, repo_name: str) -> Optional[dict]:
        """Load cached metrics for *repo_name*, or return None if absent."""
        cache_file = self._cache_path(repo_name)
        if cache_file.exists():
            try:
                with cache_file.open("r", encoding="utf-8") as fh:
                    return json.load(fh)
            except Exception:
                logger.debug("Corrupt cache for %s – recomputing.", repo_name)
        return None

    def _save_cache(self, repo_name: str, metrics: dict) -> None:
        """Persist *metrics* to the repository's cache file."""
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = self._cache_path(repo_name)
        try:
            with cache_file.open("w", encoding="utf-8") as fh:
                json.dump(metrics, fh, indent=2)
        except Exception:
            logger.debug("Failed to write cache for %s.", repo_name)

    # ------------------------------------------------------------------
    # CSV helpers
    # ------------------------------------------------------------------

    def _load_processed_names(self, csv_path: Path) -> Set[str]:
        """Return the set of repository names already present in *csv_path*."""
        if not csv_path.exists():
            return set()
        try:
            existing = pd.read_csv(csv_path, usecols=["repository_name"])
            return set(existing["repository_name"].dropna().astype(str))
        except Exception:
            logger.debug("Could not read existing CSV at %s.", csv_path)
            return set()

    def _append_row_to_csv(self, row: dict, csv_path: Path) -> None:
        """Append a single *row* dict to *csv_path*, writing the header only once."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_row = pd.DataFrame([row])
        write_header = not csv_path.exists()
        df_row.to_csv(csv_path, mode="a", header=write_header, index=False)

    # ------------------------------------------------------------------
    # Core extraction
    # ------------------------------------------------------------------

    def extract_repository_features(self, repository_path: Path) -> dict:
        """Extract features for a single repository.

        Checks the on-disk cache first; computes and caches on a miss.

        Args:
            repository_path: Path to the repository.

        Returns:
            A dictionary containing the extracted features.
        """
        repo_name = repository_path.name

        # --- cache hit ---
        cached = self._load_cache(repo_name)
        if cached is not None:
            logger.info("Loaded cached repository %s", repo_name)
            return cached

        # --- cache miss: compute ---
        logger.info("Extracting features from repository %s", repo_name)
        try:
            metrics_extractor = MetricsExtractor(repository_path=repository_path)
            base_metrics = metrics_extractor.extract_metrics()

            complexity_scorer = ComplexityScorer()
            complexity_metrics = complexity_scorer.score_repository(repo_path=repository_path)

            merged = {**base_metrics, **complexity_metrics}
            merged["repository_name"] = repo_name

            self._save_cache(repo_name, merged)
            return merged
        except Exception as e:
            logger.error(
                "Failed to extract features for %s",
                repo_name,
                exc_info=True,
            )
            raise e

    # ------------------------------------------------------------------
    # Dataset build
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        output_csv: Path = _OUTPUT_CSV,
    ) -> pd.DataFrame:
        """Build the dataset by iterating over all repositories.

        Supports resume: repositories already written to *output_csv* are
        skipped.  Each successful repository is appended to *output_csv*
        immediately so progress survives interruptions.

        Args:
            output_csv: Incremental save target (default: data/datasets/repositories.csv).

        Returns:
            A pandas DataFrame containing all features for all successfully
            processed repositories (existing rows + newly computed rows).
        """
        logger.info(
            "Building dataset from repositories in %s",
            self.repositories_dir,
        )

        if not self.repositories_dir.exists():
            logger.warning(
                "Repositories directory does not exist: %s",
                self.repositories_dir,
            )
            self._dataset = pd.DataFrame()
            return self._dataset

        # Resume support: discover already-processed repositories
        processed_names: Set[str] = self._load_processed_names(output_csv)
        if processed_names:
            logger.info(
                "Resuming – %d repositories already in CSV, skipping them.",
                len(processed_names),
            )

        all_repo_paths = [p for p in self.repositories_dir.iterdir() if p.is_dir()]
        total = len(all_repo_paths)
        new_rows: list[dict] = []

        for idx, repo_path in enumerate(all_repo_paths, start=1):
            repo_name = repo_path.name

            # Skip already-processed repositories
            if repo_name in processed_names:
                logger.info(
                    "Skipping already-processed repository %s (%d/%d)",
                    repo_name,
                    idx,
                    total,
                )
                continue

            logger.info("Processing repository %d/%d: %s", idx, total, repo_name)

            try:
                features = self.extract_repository_features(repo_path)
                new_rows.append(features)

                # Incremental save – write immediately, no duplicates
                self._append_row_to_csv(features, output_csv)
                row_count = len(processed_names) + len(new_rows)
                logger.info(
                    "Completed repository %s. Dataset rows: %d",
                    repo_name,
                    row_count,
                )
            except Exception:
                logger.error(
                    "Skipping repository %s due to failure.",
                    repo_name,
                    exc_info=True,
                )
                continue

        # Build in-memory dataset from the full CSV (existing + new)
        if output_csv.exists():
            try:
                self._dataset = pd.read_csv(output_csv)
            except Exception:
                self._dataset = pd.DataFrame(new_rows)
        else:
            self._dataset = pd.DataFrame(new_rows)

        logger.info(
            "Successfully built dataset with %d total rows.",
            len(self._dataset),
        )
        return self._dataset

    # ------------------------------------------------------------------
    # I/O helpers (preserved interface)
    # ------------------------------------------------------------------

    def save_csv(self, output_path: Path) -> None:
        """Save the dataset to a CSV file.

        Args:
            output_path: Path to the output CSV file.
        """
        logger.info("Saving dataset to CSV: %s", output_path)
        try:
            if self._dataset is None:
                logger.warning(
                    "Dataset is empty. Call build_dataset() first.",
                    extra={"output_path": str(output_path)},
                )
                return

            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._dataset.to_csv(output_path, index=False)
            logger.info("Successfully saved dataset to %s", output_path)
        except Exception:
            logger.error(
                "Failed to save dataset to CSV: %s",
                output_path,
                exc_info=True,
            )

    def load_csv(self, input_path: Path) -> pd.DataFrame:
        """Load a dataset from a CSV file.

        Args:
            input_path: Path to the input CSV file.

        Returns:
            A pandas DataFrame containing the loaded dataset.
        """
        logger.info("Loading dataset from CSV: %s", input_path)
        try:
            self._dataset = pd.read_csv(input_path)
            logger.info("Successfully loaded dataset (%d rows)", len(self._dataset))
            return self._dataset
        except Exception as e:
            logger.error(
                "Failed to load dataset from CSV: %s",
                input_path,
                exc_info=True,
            )
            raise e


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    builder = DatasetBuilder()
    df = builder.build_dataset()

    print(df.head())
    print("Dataset shape:", df.shape)

    output_csv = Path("data/datasets/repositories.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    builder.save_csv(output_csv)

    print(f"Successfully generated dataset at {output_csv}")
