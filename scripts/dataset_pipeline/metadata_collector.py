"""GitHub metadata collection stage for the RepoGraph AI dataset pipeline.

Reads repository candidates from CSV, enriches them with GitHub REST API
metadata, and writes a resumable metadata dataset.
"""

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests
from tqdm import tqdm

try:
    from scripts.dataset_pipeline.config import GITHUB_TOKEN
    from scripts.dataset_pipeline.collection_state import CollectionStateManager, STATE_FILE_PATH
    from scripts.dataset_pipeline.utils import setup_pipeline_logging
except ImportError:
    from config import GITHUB_TOKEN
    from collection_state import CollectionStateManager, STATE_FILE_PATH
    from utils import setup_pipeline_logging

logger = setup_pipeline_logging(__name__)

CANDIDATES_CSV_PATH = Path("data/datasets/repository_candidates.csv")
METADATA_CSV_PATH = Path("data/datasets/repository_metadata.csv")
METADATA_FAILURES_CSV_PATH = Path("data/logs/metadata_failures.csv")
METADATA_CACHE_DIR = Path("data/dataset_pipeline/metadata_cache")
GITHUB_API_URL = "https://api.github.com"
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0
REQUEST_TIMEOUT_SECONDS = 20


class GitHubAPIClient:
    """Reusable GitHub REST API client with retries, cache, and rate-limit handling."""

    def __init__(
        self,
        token: Optional[str] = None,
        cache_dir: Path = METADATA_CACHE_DIR,
        max_retries: int = MAX_RETRIES,
    ):
        """Initializes the GitHub API client.

        Args:
            token: Optional GitHub personal access token.
            cache_dir: Local directory used to cache JSON API responses.
            max_retries: Number of request attempts for transient failures.
        """
        self.token = token or GITHUB_TOKEN or os.getenv("GITHUB_TOKEN", "")
        self.cache_dir = Path(cache_dir)
        self.max_retries = max_retries
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "RepoGraph-AI-Metadata-Collector",
        }
        if self.token:
            self.headers["Authorization"] = f"Bearer {self.token}"
            logger.info("GitHub metadata client initialized with token authentication.")
        else:
            logger.warning("No GitHub token configured. Metadata collection may hit low rate limits.")

    def cache_path_for_url(self, url: str) -> Path:
        """Returns a deterministic cache file path for an API URL."""
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def load_cached_response(self, url: str) -> Optional[Any]:
        """Loads a cached JSON response for an API URL when present."""
        cache_path = self.cache_path_for_url(url)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read metadata cache %s: %s", cache_path, exc)
            return None

    def save_cached_response(self, url: str, data: Any) -> None:
        """Saves a JSON API response to the local cache."""
        cache_path = self.cache_path_for_url(url)
        try:
            with open(cache_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
        except OSError as exc:
            logger.warning("Failed to write metadata cache %s: %s", cache_path, exc)

    def sleep_for_rate_limit(self, response: requests.Response) -> None:
        """Sleeps until GitHub's rate-limit reset when reset metadata is available."""
        reset_header = response.headers.get("X-RateLimit-Reset")
        if not reset_header:
            time.sleep(60)
            return

        try:
            reset_epoch = int(reset_header)
        except ValueError:
            time.sleep(60)
            return

        sleep_seconds = max(reset_epoch - int(time.time()) + 5, 5)
        logger.warning("GitHub rate limit reached. Sleeping for %s seconds.", sleep_seconds)
        time.sleep(sleep_seconds)

    def get_json(self, endpoint: str, use_cache: bool = True) -> Any:
        """Fetches JSON from a GitHub REST endpoint with cache and retries.

        Args:
            endpoint: Absolute URL or API path beginning with ``/``.
            use_cache: Whether to read/write the local response cache.
        """
        url = endpoint if endpoint.startswith("http") else f"{GITHUB_API_URL}{endpoint}"
        if use_cache:
            cached_response = self.load_cached_response(url)
            if cached_response is not None:
                return cached_response

        last_error: Optional[str] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.get(url, headers=self.headers, timeout=REQUEST_TIMEOUT_SECONDS)
            except requests.RequestException as exc:
                last_error = str(exc)
                logger.warning("Request error for %s on attempt %s: %s", url, attempt, exc)
                if attempt < self.max_retries:
                    time.sleep(BACKOFF_FACTOR ** attempt)
                continue

            if response.status_code == 200:
                data = response.json()
                if use_cache:
                    self.save_cached_response(url, data)
                return data

            if response.status_code in (403, 429):
                last_error = f"Rate limited: HTTP {response.status_code}"
                self.sleep_for_rate_limit(response)
                continue

            if response.status_code in (500, 502, 503, 504):
                last_error = f"Transient server error: HTTP {response.status_code}"
                logger.warning("%s for %s on attempt %s", last_error, url, attempt)
                if attempt < self.max_retries:
                    time.sleep(BACKOFF_FACTOR ** attempt)
                continue

            last_error = f"HTTP {response.status_code}: {response.text[:500]}"
            break

        raise RuntimeError(last_error or f"Failed to fetch GitHub endpoint: {url}")

    def get_paginated_json(self, endpoint: str, per_page: int = 100) -> List[Any]:
        """Fetches all list items from a paginated GitHub REST endpoint."""
        all_items: List[Any] = []
        page = 1
        separator = "&" if "?" in endpoint else "?"

        while True:
            page_endpoint = f"{endpoint}{separator}per_page={per_page}&page={page}"
            data = self.get_json(page_endpoint)
            if not isinstance(data, list) or not data:
                break

            all_items.extend(data)
            if len(data) < per_page:
                break
            page += 1

        return all_items


class MetadataCollector:
    """Collects GitHub repository metadata and writes a CSV dataset."""

    def __init__(
        self,
        candidates_csv_path: Path = CANDIDATES_CSV_PATH,
        output_csv_path: Path = METADATA_CSV_PATH,
        failures_csv_path: Path = METADATA_FAILURES_CSV_PATH,
        client: Optional[GitHubAPIClient] = None,
    ):
        """Initializes metadata collection paths and API client."""
        self.candidates_csv_path = Path(candidates_csv_path)
        self.output_csv_path = Path(output_csv_path)
        self.failures_csv_path = Path(failures_csv_path)
        self.client = client or GitHubAPIClient()
        self.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
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
        """Loads repository candidates from the discovery CSV."""
        if not self.candidates_csv_path.exists():
            raise FileNotFoundError(f"Repository candidates CSV not found: {self.candidates_csv_path}")

        df = pd.read_csv(self.candidates_csv_path).fillna("")
        repositories = df.to_dict(orient="records")
        logger.info("Loaded %s repository candidates from %s", len(repositories), self.candidates_csv_path)
        return repositories

    def load_existing_metadata(self) -> Tuple[List[Dict[str, Any]], Set[Tuple[str, str]]]:
        """Loads existing metadata output to support resumable runs."""
        if not self.output_csv_path.exists():
            return [], set()

        df = pd.read_csv(self.output_csv_path).fillna("")
        records = df.to_dict(orient="records")
        processed = {
            (str(record.get("owner", "")).lower(), str(record.get("repository_name", "")).lower())
            for record in records
            if record.get("owner") and record.get("repository_name")
        }
        logger.info("Loaded %s existing metadata records from %s", len(records), self.output_csv_path)
        return records, processed

    def get_repository_identity(self, repo: Dict[str, Any]) -> Tuple[str, str]:
        """Extracts owner and repository name from a candidate row."""
        owner = str(repo.get("owner", "")).strip()
        repository_name = str(repo.get("repository_name", "")).strip()

        if owner and repository_name:
            return owner, repository_name

        full_name = str(repo.get("full_name", "")).strip()
        if "/" in full_name:
            owner_part, repo_part = full_name.split("/", 1)
            return owner_part.strip(), repo_part.strip()

        html_url = str(repo.get("html_url", "")).strip().rstrip("/")
        if "github.com/" in html_url:
            path = html_url.split("github.com/", 1)[1]
            parts = path.split("/")
            if len(parts) >= 2:
                return parts[0].strip(), parts[1].strip()

        return "", ""

    def count_paginated_endpoint(self, endpoint: str) -> int:
        """Counts all items from a paginated GitHub endpoint."""
        return len(self.client.get_paginated_json(endpoint))

    def count_open_pull_requests(self, owner: str, repository_name: str) -> int:
        """Counts currently open pull requests for a repository."""
        endpoint = f"/repos/{owner}/{repository_name}/pulls?state=open"
        return len(self.client.get_paginated_json(endpoint))

    def collect_repository_metadata(self, owner: str, repository_name: str) -> Dict[str, Any]:
        """Collects repository, language, contributor, release, and pull-request metadata."""
        repo_data = self.client.get_json(f"/repos/{owner}/{repository_name}")
        languages = self.client.get_json(f"/repos/{owner}/{repository_name}/languages")
        contributors = self.count_paginated_endpoint(f"/repos/{owner}/{repository_name}/contributors")
        releases = self.count_paginated_endpoint(f"/repos/{owner}/{repository_name}/releases")
        open_pull_requests = self.count_open_pull_requests(owner, repository_name)
        license_data = repo_data.get("license") or {}

        return {
            "owner": owner,
            "repository_name": repository_name,
            "html_url": repo_data.get("html_url", ""),
            "description": repo_data.get("description", ""),
            "primary_language": repo_data.get("language", ""),
            "languages": json.dumps(languages, ensure_ascii=False, sort_keys=True),
            "topics": ",".join(repo_data.get("topics") or []),
            "license": license_data.get("spdx_id") or license_data.get("key") or license_data.get("name") or "",
            "stars": repo_data.get("stargazers_count", 0),
            "forks": repo_data.get("forks_count", 0),
            "watchers": repo_data.get("watchers_count", 0),
            "subscribers": repo_data.get("subscribers_count", 0),
            "open_issues": repo_data.get("open_issues_count", 0),
            "default_branch": repo_data.get("default_branch", ""),
            "created_at": repo_data.get("created_at", ""),
            "updated_at": repo_data.get("updated_at", ""),
            "pushed_at": repo_data.get("pushed_at", ""),
            "size": repo_data.get("size", 0),
            "archived": repo_data.get("archived", False),
            "disabled": repo_data.get("disabled", False),
            "has_wiki": repo_data.get("has_wiki", False),
            "has_projects": repo_data.get("has_projects", False),
            "has_downloads": repo_data.get("has_downloads", False),
            "homepage": repo_data.get("homepage", ""),
            "visibility": repo_data.get("visibility", ""),
            "forks_count": repo_data.get("forks_count", 0),
            "network_count": repo_data.get("network_count", 0),
            "open_pull_requests": open_pull_requests,
            "contributor_count": contributors,
            "release_count": releases,
        }

    def save_metadata(self, records: List[Dict[str, Any]]) -> None:
        """Persists metadata records to the output CSV."""
        pd.DataFrame(records).to_csv(self.output_csv_path, index=False, encoding="utf-8")

    def save_failures(self, failures: List[Dict[str, str]]) -> None:
        """Persists metadata collection failures to the failure CSV."""
        failure_df = pd.DataFrame(failures, columns=["repository", "url", "reason"])
        failure_df.to_csv(self.failures_csv_path, index=False, encoding="utf-8")
        logger.info("Wrote %s metadata failures to %s", len(failures), self.failures_csv_path)

    def run(self) -> Dict[str, Any]:
        """Runs metadata collection from candidates CSV to metadata CSV."""
        candidates = self.load_candidates()
        records, processed = self.load_existing_metadata()
        completed = len(self.state_manager.get_completed_repositories("metadata_collected"))
        logger.info("Resume position for metadata: %s repositories already marked collected.", completed)
        failures: List[Dict[str, str]] = []
        successful = 0
        skipped = 0

        for candidate in tqdm(candidates, desc="Collecting metadata"):
            owner, repository_name = self.get_repository_identity(candidate)
            repository = f"{owner}/{repository_name}" if owner and repository_name else "unknown"
            url = str(candidate.get("html_url", "")).strip()

            if not owner or not repository_name:
                self.state_manager.mark_failed(repository, "Missing owner or repository name")
                logger.warning("Failed repository state updated for %s", repository)
                failures.append({
                    "repository": repository,
                    "url": url,
                    "reason": "Missing owner or repository name",
                })
                continue

            dedupe_key = (owner.lower(), repository_name.lower())
            state_record = self.state_manager.get_repository(repository)
            if state_record and state_record.get("metadata_collected", False):
                logger.info("Skipping completed metadata collection from state: %s", repository)
                skipped += 1
                continue

            if dedupe_key in processed:
                skipped += 1
                self.state_manager.update_step(repository, "metadata_collected", True)
                logger.info("Updated collection state from existing metadata CSV for %s", repository)
                continue

            try:
                metadata = self.collect_repository_metadata(owner, repository_name)
                records.append(metadata)
                processed.add(dedupe_key)
                successful += 1
                self.save_metadata(records)
                self.state_manager.update_step(repository, "metadata_collected", True)
                logger.info("Updated collection state for metadata collection: %s", repository)
            except Exception as exc:
                logger.error("Failed to collect metadata for %s: %s", repository, exc)
                self.state_manager.mark_failed(repository, str(exc))
                logger.warning("Failed repository state updated for %s", repository)
                failures.append({
                    "repository": repository,
                    "url": url,
                    "reason": str(exc),
                })

        self.save_metadata(records)
        self.save_failures(failures)

        output_df = pd.DataFrame(records)
        average_stars = round(float(output_df["stars"].mean()), 2) if not output_df.empty else 0.0
        average_forks = round(float(output_df["forks"].mean()), 2) if not output_df.empty else 0.0
        summary = {
            "repositories_processed": len(candidates),
            "successful": successful + skipped,
            "failed": len(failures),
            "average_stars": average_stars,
            "average_forks": average_forks,
        }

        logger.info("Metadata collection summary:")
        logger.info("Repositories processed: %s", summary["repositories_processed"])
        logger.info("Successful: %s", summary["successful"])
        logger.info("Failed: %s", summary["failed"])
        logger.info("Average stars: %s", summary["average_stars"])
        logger.info("Average forks: %s", summary["average_forks"])

        print("Metadata collection summary")
        print(f"Repositories processed: {summary['repositories_processed']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Average stars: {summary['average_stars']}")
        print(f"Average forks: {summary['average_forks']}")

        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    collector = MetadataCollector()
    collector.run()
