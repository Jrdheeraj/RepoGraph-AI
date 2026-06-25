"""Collection state manager for the RepoGraph AI dataset pipeline.

Tracks each repository's progress through discovery, cloning, metadata,
metrics, and labeling stages so interrupted runs can safely resume.
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

try:
    from scripts.dataset_pipeline.utils import setup_pipeline_logging
except ImportError:
    from utils import setup_pipeline_logging

logger = setup_pipeline_logging(__name__)

STATE_DIR = Path("data/state")
STATE_FILE_PATH = STATE_DIR / "collection_state.json"

STEP_FIELDS = {
    "discovered",
    "cloned",
    "metadata_collected",
    "metrics_extracted",
    "labels_generated",
}


class CollectionStateManager:
    """Manages persistent repository pipeline state with atomic JSON writes."""

    def __init__(self, state_file_path: Path = STATE_FILE_PATH):
        """Initializes the manager and loads or creates the state file.

        Args:
            state_file_path: JSON file used to persist collection state.
        """
        self.state_file_path = Path(state_file_path)
        self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.state: Dict[str, Dict[str, Any]] = {}
        self.load_state()

    def _empty_record(self, repo_name: str) -> Dict[str, Any]:
        """Creates a default state record for a repository."""
        return {
            "repository_name": repo_name,
            "discovered": False,
            "cloned": False,
            "metadata_collected": False,
            "metrics_extracted": False,
            "labels_generated": False,
            "failed": False,
            "last_step": "",
            "last_updated": "",
        }

    def _timestamp(self) -> str:
        """Returns the current UTC timestamp in ISO-8601 format."""
        return datetime.now(timezone.utc).isoformat()

    def _normalize_repo_name(self, repo_name: str) -> str:
        """Normalizes repository names used as state keys."""
        return str(repo_name).strip()

    def _repo_name_from_row(self, row: Dict[str, Any]) -> str:
        """Extracts a repository name from a candidate CSV row."""
        owner = str(row.get("owner", "")).strip()
        repository_name = str(row.get("repository_name", "")).strip()
        if owner and repository_name:
            return f"{owner}/{repository_name}"

        full_name = str(row.get("full_name", "")).strip()
        if full_name:
            return full_name

        html_url = str(row.get("html_url", "")).strip().rstrip("/")
        if "github.com/" in html_url:
            parts = html_url.split("github.com/", 1)[1].split("/")
            if len(parts) >= 2:
                return f"{parts[0]}/{parts[1]}"

        return repository_name

    def initialize_state(self, candidate_csv: Path) -> None:
        """Initializes or updates state records from a repository candidate CSV.

        Existing repository records are preserved to support safe resume behavior.
        New repositories are added and marked as discovered.
        """
        candidate_csv = Path(candidate_csv)
        if not candidate_csv.exists():
            raise FileNotFoundError(f"Candidate CSV not found: {candidate_csv}")

        df = pd.read_csv(candidate_csv).fillna("")
        added_count = 0

        with self._lock:
            for row in df.to_dict(orient="records"):
                repo_name = self._normalize_repo_name(self._repo_name_from_row(row))
                if not repo_name:
                    continue

                if repo_name not in self.state:
                    self.state[repo_name] = self._empty_record(repo_name)
                    added_count += 1

                record = self.state[repo_name]
                record["discovered"] = True
                record["last_step"] = "discovered"
                record["last_updated"] = self._timestamp()

            self.save_state()

        logger.info("Initialized collection state from %s; added %s repositories.", candidate_csv, added_count)

    def repository_exists(self, repo_name: str) -> bool:
        """Returns True when a repository is present in the state file."""
        key = self._normalize_repo_name(repo_name)
        with self._lock:
            return key in self.state

    def get_repository(self, repo_name: str) -> Optional[Dict[str, Any]]:
        """Returns a repository state record, or None when absent."""
        key = self._normalize_repo_name(repo_name)
        with self._lock:
            record = self.state.get(key)
            return dict(record) if record else None

    def update_step(self, repo_name: str, step_name: str, status: bool) -> None:
        """Updates a pipeline step status for a repository.

        Args:
            repo_name: Repository name, usually ``owner/repository``.
            step_name: One of the supported step fields.
            status: Completion status for the step.
        """
        if step_name not in STEP_FIELDS:
            raise ValueError(f"Unsupported step name: {step_name}")

        key = self._normalize_repo_name(repo_name)
        if not key:
            raise ValueError("Repository name cannot be empty.")

        with self._lock:
            if key not in self.state:
                self.state[key] = self._empty_record(key)

            record = self.state[key]
            record[step_name] = bool(status)
            record["last_step"] = step_name
            record["last_updated"] = self._timestamp()
            if status:
                record["failed"] = False
            self.save_state()

        logger.info("Updated %s step '%s' to %s.", key, step_name, status)

    def mark_failed(self, repo_name: str, reason: str) -> None:
        """Marks a repository as failed and logs the failure reason."""
        key = self._normalize_repo_name(repo_name)
        if not key:
            raise ValueError("Repository name cannot be empty.")

        with self._lock:
            if key not in self.state:
                self.state[key] = self._empty_record(key)

            record = self.state[key]
            record["failed"] = True
            record["last_step"] = "failed"
            record["last_updated"] = self._timestamp()
            self.save_state()

        logger.warning("Marked repository as failed: %s; reason=%s", key, reason)

    def get_pending_repositories(self, step_name: str) -> List[Dict[str, Any]]:
        """Returns non-failed repositories that have not completed a step."""
        if step_name not in STEP_FIELDS:
            raise ValueError(f"Unsupported step name: {step_name}")

        with self._lock:
            return [
                dict(record)
                for record in self.state.values()
                if not record.get(step_name, False) and not record.get("failed", False)
            ]

    def get_completed_repositories(self, step_name: str) -> List[Dict[str, Any]]:
        """Returns repositories that have completed a step."""
        if step_name not in STEP_FIELDS:
            raise ValueError(f"Unsupported step name: {step_name}")

        with self._lock:
            return [dict(record) for record in self.state.values() if record.get(step_name, False)]

    def get_failed_repositories(self) -> List[Dict[str, Any]]:
        """Returns repositories currently marked as failed."""
        with self._lock:
            return [dict(record) for record in self.state.values() if record.get("failed", False)]

    def save_state(self) -> None:
        """Atomically saves state to disk using a temporary file and replace."""
        with self._lock:
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                prefix=f"{self.state_file_path.name}.",
                suffix=".tmp",
                dir=str(self.state_file_path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as temp_file:
                    json.dump(self.state, temp_file, indent=2, ensure_ascii=False, sort_keys=True)
                    temp_file.write("\n")
                    temp_file.flush()
                    os.fsync(temp_file.fileno())
                os.replace(temp_path, self.state_file_path)
            except Exception:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise

    def load_state(self) -> None:
        """Loads state from disk, creating an empty state file when absent."""
        with self._lock:
            if not self.state_file_path.exists():
                self.state = {}
                self.save_state()
                logger.info("Created new collection state file at %s", self.state_file_path)
                return

            try:
                with open(self.state_file_path, "r", encoding="utf-8") as state_file:
                    loaded_state = json.load(state_file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Collection state file is not valid JSON: {self.state_file_path}") from exc

            if not isinstance(loaded_state, dict):
                raise ValueError(f"Collection state must be a JSON object: {self.state_file_path}")

            self.state = {
                self._normalize_repo_name(repo_name): record
                for repo_name, record in loaded_state.items()
                if isinstance(record, dict)
            }
            logger.info("Loaded collection state for %s repositories.", len(self.state))

    def reset_repository(self, repo_name: str) -> None:
        """Resets a repository to the default unprocessed state."""
        key = self._normalize_repo_name(repo_name)
        if not key:
            raise ValueError("Repository name cannot be empty.")

        with self._lock:
            self.state[key] = self._empty_record(key)
            self.state[key]["last_updated"] = self._timestamp()
            self.save_state()

        logger.info("Reset collection state for repository: %s", key)

    def export_summary(self) -> Dict[str, Any]:
        """Returns aggregate counts for all tracked pipeline steps."""
        with self._lock:
            total = len(self.state)
            summary: Dict[str, Any] = {
                "total_repositories": total,
                "failed": sum(1 for record in self.state.values() if record.get("failed", False)),
            }
            for step_name in sorted(STEP_FIELDS):
                completed = sum(1 for record in self.state.values() if record.get(step_name, False))
                summary[f"{step_name}_completed"] = completed
                summary[f"{step_name}_pending"] = total - completed

        logger.info("Collection state summary: %s", summary)
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = CollectionStateManager()
    print(json.dumps(manager.export_summary(), indent=2))
