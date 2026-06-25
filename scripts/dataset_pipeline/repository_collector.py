"""Repository cloning stage for the RepoGraph AI dataset pipeline.

Reads repository candidates from CSV, clones each repository with GitPython,
and records clone failures without stopping the batch.
"""

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from git import GitCommandError, Repo
from tqdm import tqdm

try:
    from scripts.dataset_pipeline.config import CLONE_TIMEOUT_SECONDS
    from scripts.dataset_pipeline.collection_state import CollectionStateManager, STATE_FILE_PATH
    from scripts.dataset_pipeline.utils import setup_pipeline_logging
except ImportError:
    from config import CLONE_TIMEOUT_SECONDS
    from collection_state import CollectionStateManager, STATE_FILE_PATH
    from utils import setup_pipeline_logging

logger = setup_pipeline_logging(__name__)

CANDIDATES_CSV_PATH = Path("data/datasets/repository_candidates.csv")
REPOSITORIES_DIR = Path("data/repositories")
CLONE_FAILURES_CSV_PATH = Path("data/logs/clone_failures.csv")
MAX_CLONE_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0


class RepositoryCollector:
    """Clones candidate repositories into a local repository directory."""

    def __init__(
        self,
        candidates_csv_path: Path = CANDIDATES_CSV_PATH,
        output_root: Path = REPOSITORIES_DIR,
        failures_csv_path: Path = CLONE_FAILURES_CSV_PATH,
        max_retries: int = MAX_CLONE_RETRIES,
    ):
        """Initializes the repository collector.

        Args:
            candidates_csv_path: CSV file produced by the discovery stage.
            output_root: Local directory where repositories are cloned.
            failures_csv_path: CSV file where clone failures are written.
            max_retries: Number of clone attempts per repository.
        """
        self.candidates_csv_path = Path(candidates_csv_path)
        self.output_root = Path(output_root)
        self.failures_csv_path = Path(failures_csv_path)
        self.max_retries = max_retries
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.failures_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_manager = self.initialize_state_manager()

    def initialize_state_manager(self) -> CollectionStateManager:
        """Loads collection state and initializes it from candidates on first run."""
        state_exists = STATE_FILE_PATH.exists()
        manager = CollectionStateManager()
        if not state_exists and self.candidates_csv_path.exists():
            manager.initialize_state(self.candidates_csv_path)
            logger.info("Initialized collection state from %s", self.candidates_csv_path)
        logger.info("Loaded collection state: %s", manager.export_summary())
        return manager

    def load_candidates(self) -> List[Dict[str, Any]]:
        """Loads candidate repositories from the discovery CSV."""
        if not self.candidates_csv_path.exists():
            raise FileNotFoundError(f"Repository candidates CSV not found: {self.candidates_csv_path}")

        df = pd.read_csv(self.candidates_csv_path).fillna("")
        repositories = df.to_dict(orient="records")
        logger.info("Loaded %s repository candidates from %s", len(repositories), self.candidates_csv_path)
        return repositories

    def get_repository_identity(self, repo: Dict[str, Any]) -> Tuple[str, str, str]:
        """Extracts repository display name, clone URL, and default branch."""
        owner = str(repo.get("owner", "")).strip()
        repository_name = str(repo.get("repository_name", "")).strip()
        html_url = str(repo.get("html_url", "")).strip()
        default_branch = str(repo.get("default_branch", "")).strip()

        if owner and repository_name:
            repository = f"{owner}/{repository_name}"
        else:
            repository = str(repo.get("full_name", "")).strip()

        if not html_url and repository:
            html_url = f"https://github.com/{repository}"

        return repository, html_url, default_branch

    def get_target_path(self, repository: str) -> Path:
        """Returns the local clone path for a GitHub owner/repository name."""
        safe_name = repository.replace("/", "__").replace("\\", "__")
        return self.output_root / safe_name

    def is_existing_clone(self, target_path: Path) -> bool:
        """Checks whether a repository has already been cloned locally."""
        return target_path.exists() and (target_path / ".git").exists()

    def remove_partial_clone(self, target_path: Path) -> None:
        """Removes an incomplete clone directory within the configured output root."""
        resolved_output_root = self.output_root.resolve()
        resolved_target = target_path.resolve()

        if resolved_target == resolved_output_root or resolved_output_root not in resolved_target.parents:
            raise ValueError(f"Refusing to remove path outside repository output root: {target_path}")

        if target_path.exists():
            shutil.rmtree(target_path)

    def clone_repository(self, repo_url: str, repository: str, default_branch: str = "") -> str:
        """Clones one repository with shallow, single-branch Git options.

        Args:
            repo_url: HTTPS URL for the repository.
            repository: GitHub owner/repository display name.
            default_branch: Default branch to clone.

        Returns:
            A status string: ``cloned`` or ``already_existed``.
        """
        target_path = self.get_target_path(repository)
        if self.is_existing_clone(target_path):
            logger.info("Repository already exists locally: %s", target_path)
            return "already_existed"

        if target_path.exists():
            logger.warning("Removing incomplete repository directory before clone: %s", target_path)
            self.remove_partial_clone(target_path)

        clone_options = ["--depth=1", "--single-branch"]
        if default_branch:
            clone_options.append(f"--branch={default_branch}")

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info("Cloning %s into %s (attempt %s/%s)", repository, target_path, attempt, self.max_retries)
                Repo.clone_from(
                    repo_url,
                    target_path,
                    multi_options=clone_options,
                    kill_after_timeout=CLONE_TIMEOUT_SECONDS,
                )
                return "cloned"
            except (GitCommandError, OSError) as exc:
                last_error = exc
                logger.warning("Clone failed for %s on attempt %s: %s", repository, attempt, exc)
                if target_path.exists():
                    self.remove_partial_clone(target_path)
                if attempt < self.max_retries:
                    time.sleep(RETRY_BACKOFF_SECONDS ** attempt)

        reason = str(last_error) if last_error else "Unknown Git clone failure"
        raise RuntimeError(reason)

    def write_failures(self, failures: List[Dict[str, str]]) -> None:
        """Writes clone failures to disk as a CSV file."""
        failure_df = pd.DataFrame(failures, columns=["repository", "url", "reason"])
        failure_df.to_csv(self.failures_csv_path, index=False, encoding="utf-8")
        logger.info("Wrote %s clone failures to %s", len(failures), self.failures_csv_path)

    def collect_all(self, repositories: List[Dict[str, Any]]) -> Dict[str, int]:
        """Clones all repositories and continues through individual failures."""
        summary = {
            "total": len(repositories),
            "successfully_cloned": 0,
            "already_existed": 0,
            "failed": 0,
        }
        failures: List[Dict[str, str]] = []

        for repo in tqdm(repositories, desc="Cloning repositories"):
            repository, url, default_branch = self.get_repository_identity(repo)
            if not repository or not url:
                summary["failed"] += 1
                failed_repository = repository or "unknown"
                self.state_manager.mark_failed(failed_repository, "Missing repository identity or URL")
                logger.warning("Failed repository state updated for %s", failed_repository)
                failures.append({
                    "repository": failed_repository,
                    "url": url,
                    "reason": "Missing repository identity or URL",
                })
                continue

            state_record = self.state_manager.get_repository(repository)
            if state_record and state_record.get("cloned", False):
                logger.info("Skipping completed repository clone from state: %s", repository)
                summary["already_existed"] += 1
                continue

            try:
                status = self.clone_repository(url, repository, default_branch)
                if status == "already_existed":
                    summary["already_existed"] += 1
                else:
                    summary["successfully_cloned"] += 1
                self.state_manager.update_step(repository, "cloned", True)
                logger.info("Updated collection state for cloned repository: %s", repository)
            except Exception as exc:
                summary["failed"] += 1
                logger.error("Failed to clone %s: %s", repository, exc)
                self.state_manager.mark_failed(repository, str(exc))
                logger.warning("Failed repository state updated for %s", repository)
                failures.append({
                    "repository": repository,
                    "url": url,
                    "reason": str(exc),
                })

        self.write_failures(failures)
        return summary

    def run(self) -> Dict[str, int]:
        """Runs the repository collection stage from CSV input to clone summary."""
        repositories = self.load_candidates()
        completed = len(self.state_manager.get_completed_repositories("cloned"))
        logger.info("Resume position for cloning: %s repositories already marked cloned.", completed)
        summary = self.collect_all(repositories)

        logger.info("Repository cloning summary:")
        logger.info("Total repositories: %s", summary["total"])
        logger.info("Successfully cloned: %s", summary["successfully_cloned"])
        logger.info("Already existed: %s", summary["already_existed"])
        logger.info("Failed: %s", summary["failed"])

        print("Repository cloning summary")
        print(f"Total repositories: {summary['total']}")
        print(f"Successfully cloned: {summary['successfully_cloned']}")
        print(f"Already existed: {summary['already_existed']}")
        print(f"Failed: {summary['failed']}")

        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = RepositoryCollector()
    collector.run()
