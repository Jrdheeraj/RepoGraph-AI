"""GitHub Dataset Collector for RepoGraph AI.

Responsibilities
----------------
1. Clone real-world GitHub repositories for ML training  (original, unchanged).
2. Fetch rich external metadata for every repository via the GitHub REST
   API v3 — used downstream for leakage-free label generation.

Metadata collected
------------------
  stars, forks, watchers, open_issues, closed_issues (placeholder),
  contributors_count, default_branch, license, primary_language,
  topics, created_at, updated_at, pushed_at,
  latest_release_date, latest_release_tag,
  repo_age_days, days_since_last_commit

Repository maturity flags (file-tree detection)
------------------------------------------------
  has_tests, has_docs, has_ci, has_dockerfile,
  has_type_hints, has_examples, has_benchmarks

Authentication
--------------
Set GITHUB_TOKEN in the environment to raise the rate-limit from
60 (unauthenticated) to 5 000 requests/hour (authenticated).

    $env:GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"   # PowerShell
    export GITHUB_TOKEN="ghp_xxxxxxxxxxxx"   # bash / zsh

Resilience guarantees
---------------------
* 401 Unauthorized  → strips the bad token, retries anonymously (no looping).
* 403 + rate-limit  → sleeps until reset + 5 s, retries automatically.
* 404 Not Found     → returns None immediately (no wasted retries).
* Network error     → exponential backoff (1 s, 2 s, 4 s, 8 s); max 4 attempts.
* Trees API failure → all maturity flags default to False; repo is NOT skipped.
* Any single repo failure → logged as WARNING; collection continues.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import git
import requests
from requests.exceptions import ConnectionError, HTTPError, Timeout

REPO_ALIASES = {
    "fastap": "tiangolo/fastapi",
    "prefec": "PrefectHQ/prefect",
    "airflow": "apache/airflow",
    "pydantic": "pydantic/pydantic",
    "ray": "ray-project/ray",
    "rich": "Textualize/rich",
    "sqlalchemy": "sqlalchemy/sqlalchemy",
    "typer": "fastapi/typer"
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GitHub REST API v3 — shared authenticated session
# ---------------------------------------------------------------------------
_GH_API = "https://api.github.com"

_SESSION = requests.Session()
_SESSION.headers.update({
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
})

# Track auth state so _gh_get() can strip it on 401 without re-checking env
_authenticated = False

_token = os.environ.get("GITHUB_TOKEN", "").strip()
if _token:
    _SESSION.headers["Authorization"] = f"Bearer {_token}"
    _authenticated = True
    logger.info(
        "GitHub token detected — using authenticated requests (5 000 req/h)."
    )
else:
    logger.warning(
        "No GITHUB_TOKEN found — using unauthenticated requests (60 req/h). "
        "Set GITHUB_TOKEN to increase the limit."
    )

# ---------------------------------------------------------------------------
# Core API helper — fully hardened
# ---------------------------------------------------------------------------

def _gh_get(url: str, **params: Any) -> Optional[Any]:
    """GET a GitHub API endpoint with complete resilience handling.

    Behaviour
    ---------
    * 401 Unauthorized:
        Logs the problem, removes the Authorization header from the shared
        session, and retries once in anonymous mode.  Does NOT loop on 401.
    * 403 + X-RateLimit-Reset:
        Reads the reset timestamp, sleeps until reset + 5 s, then retries.
        Logs remaining quota and wait time.
    * 404 Not Found:
        Returns None immediately — no retry.
    * Timeout / ConnectionError / HTTPError:
        Exponential back-off: 1 s → 2 s → 4 s → 8 s (max 4 attempts total).
    * All attempts exhausted:
        Returns None and logs an ERROR — caller decides whether to skip.

    Args:
        url:    Full GitHub API URL.
        params: Optional query-string parameters passed as keyword args.

    Returns:
        Parsed JSON object, or None on unrecoverable failure.
    """
    global _authenticated
    max_attempts = 4
    attempt = 0

    while attempt < max_attempts:
        try:
            resp = _SESSION.get(url, params=params or None, timeout=30)

            # ------ 401 Unauthorized — bad token --------------------------
            if resp.status_code == 401:
                if _authenticated:
                    logger.warning(
                        "Invalid GitHub token detected. "
                        "Falling back to unauthenticated requests."
                    )
                    _SESSION.headers.pop("Authorization", None)
                    _authenticated = False
                    # Retry immediately in anonymous mode (no backoff needed)
                    continue
                else:
                    # Already anonymous — this is an unexpected 401; give up.
                    logger.error(
                        "Received 401 in unauthenticated mode for %s — giving up.", url
                    )
                    return None

            # ------ 403 + rate-limit header --------------------------------
            if resp.status_code == 403 and "X-RateLimit-Reset" in resp.headers:
                reset_ts  = int(resp.headers["X-RateLimit-Reset"])
                remaining = resp.headers.get("X-RateLimit-Remaining", "?")
                wait_s    = max(0, reset_ts - int(time.time())) + 5
                logger.warning(
                    "GitHub rate-limited — remaining quota: %s.  "
                    "Sleeping %d s until reset (attempt %d/%d).",
                    remaining, wait_s, attempt + 1, max_attempts,
                )
                time.sleep(wait_s)
                # Do NOT increment attempt — this is a quota wait, not an error
                continue

            # ------ 404 Not Found — no retry ------------------------------
            if resp.status_code == 404:
                logger.debug("404 Not Found: %s", url)
                return None

            # ------ Any other HTTP error ----------------------------------
            resp.raise_for_status()
            return resp.json()

        except Timeout:
            backoff = 2 ** attempt
            logger.warning(
                "Request timed out for %s (attempt %d/%d) — retrying in %d s.",
                url, attempt + 1, max_attempts, backoff,
            )
            time.sleep(backoff)
            attempt += 1

        except ConnectionError as exc:
            backoff = 2 ** attempt
            logger.warning(
                "Connection error for %s (attempt %d/%d): %s — retrying in %d s.",
                url, attempt + 1, max_attempts, exc, backoff,
            )
            time.sleep(backoff)
            attempt += 1

        except HTTPError as exc:
            backoff = 2 ** attempt
            logger.warning(
                "HTTP error for %s (attempt %d/%d): %s — retrying in %d s.",
                url, attempt + 1, max_attempts, exc, backoff,
            )
            time.sleep(backoff)
            attempt += 1

        except Exception as exc:  # noqa: BLE001 — safety net
            backoff = 2 ** attempt
            logger.warning(
                "Unexpected error for %s (attempt %d/%d): %s — retrying in %d s.",
                url, attempt + 1, max_attempts, exc, backoff,
            )
            time.sleep(backoff)
            attempt += 1

    logger.error("All %d attempts failed for: %s — skipping.", max_attempts, url)
    return None


# ---------------------------------------------------------------------------
# Datetime helpers
# ---------------------------------------------------------------------------

def _parse_dt(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 datetime string from the GitHub API into UTC datetime."""
    if not dt_str:
        return None
    return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))


def _days_since(dt: Optional[datetime]) -> Optional[int]:
    """Return the number of calendar days between *dt* and UTC now."""
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).days


# ---------------------------------------------------------------------------
# Main collector class
# ---------------------------------------------------------------------------

class GitHubDatasetCollector:
    """Collect and clone GitHub repositories for the RepoGraph AI dataset.

    Scalability
    -----------
    The pipeline handles 18 repositories today and 100–300 repositories
    later without any code changes — simply extend ``get_repository_list()``.
    """

    def __init__(self, target_dir: str = "data/repositories/") -> None:
        """Initialise the collector.

        Args:
            target_dir: Local directory where cloned repositories are stored.
        """
        self.target_dir = Path(target_dir)

    # ------------------------------------------------------------------
    # Repository list — extend here to scale to 100–300 repos
    # ------------------------------------------------------------------

    def get_repository_list(self) -> List[str]:
        """Return the full list of target GitHub repository clone URLs."""
        return [
            "https://github.com/tiangolo/fastapi.git",
            "https://github.com/django/django.git",
            "https://github.com/pallets/flask.git",
            "https://github.com/psf/requests.git",
            "https://github.com/langchain-ai/langchain.git",
            "https://github.com/huggingface/transformers.git",
            "https://github.com/pytorch/pytorch.git",
            "https://github.com/tensorflow/tensorflow.git",
            "https://github.com/numpy/numpy.git",
            "https://github.com/pandas-dev/pandas.git",
            "https://github.com/scikit-learn/scikit-learn.git",
            "https://github.com/apache/airflow.git",
            "https://github.com/PrefectHQ/prefect.git",
            "https://github.com/ray-project/ray.git",
            "https://github.com/pydantic/pydantic.git",
            "https://github.com/tiangolo/typer.git",
            "https://github.com/Textualize/rich.git",
            "https://github.com/sqlalchemy/sqlalchemy.git",
        ]

    # ------------------------------------------------------------------
    # URL parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _owner_repo(repo_url: str) -> Tuple[str, str]:
        """Extract (owner, repo) from any GitHub HTTPS clone URL."""
        repo_name = repo_url.rstrip("/").removesuffix(".git").split("/")[-1].lower().strip()
        if repo_name in REPO_ALIASES:
            parts = REPO_ALIASES[repo_name].split("/")
            return parts[0], parts[1]

        slug  = repo_url.rstrip("/").removesuffix(".git").split("github.com/")[-1]
        parts = slug.split("/")
        owner = parts[0]
        repo  = parts[1] if len(parts) > 1 else ""
        return owner, repo

    @staticmethod
    def _repo_name_from_url(repo_url: str) -> str:
        """Return the bare repository name from a clone URL."""
        return repo_url.rstrip("/").removesuffix(".git").split("/")[-1]

    # ------------------------------------------------------------------
    # Maturity / quality flag detection via GitHub Trees API
    # ------------------------------------------------------------------

    def _detect_maturity_flags(self, owner: str, repo: str) -> Dict[str, bool]:
        """Detect repository quality signals from the full file-tree.

        Uses the GitHub Git Trees API (recursive=1) — single request for the
        entire tree, no directory-by-directory enumeration.

        On any API failure the flags all default to False and the repository
        is NOT skipped — metadata generation continues normally.

        Returns:
            Dict with 7 boolean keys:
            has_tests, has_docs, has_ci, has_dockerfile,
            has_type_hints, has_examples, has_benchmarks
        """
        # Safe defaults — returned on any failure
        flags: Dict[str, bool] = {
            "has_tests":      False,
            "has_docs":       False,
            "has_ci":         False,
            "has_dockerfile": False,
            "has_type_hints": False,
            "has_examples":   False,
            "has_benchmarks": False,
        }

        try:
            tree_resp = _gh_get(
                f"{_GH_API}/repos/{owner}/{repo}/git/trees/HEAD",
                recursive="1",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Trees API exception for %s/%s: %s — all flags default to False.",
                owner, repo, exc,
            )
            return flags

        if not tree_resp:
            logger.warning(
                "Could not fetch file tree for %s/%s — "
                "maturity flags defaulting to False.  Continuing.",
                owner, repo,
            )
            return flags

        # Lower-case all paths for case-insensitive matching
        paths = [item["path"].lower() for item in tree_resp.get("tree", [])]

        flags["has_tests"] = any(
            p.startswith(("tests/", "test/", "testing/"))
            or p in ("tests", "test")
            or p.endswith("_test.py")
            or p.endswith("/test.py")
            for p in paths
        )
        flags["has_docs"] = any(
            p.startswith(("docs/", "doc/", "documentation/"))
            or p in ("readme.md", "readme.rst", "readme.txt")
            for p in paths
        )
        flags["has_ci"] = any(
            p.startswith((
                ".github/workflows/",
                ".circleci/",
                ".travis",
                "jenkins",
                ".gitlab-ci",
                ".azure-pipelines",
                ".buildkite",
            ))
            or p in ("tox.ini", ".travis.yml", "jenkinsfile", "setup.cfg")
            for p in paths
        )
        flags["has_dockerfile"] = any(
            p == "dockerfile"
            or p.startswith("docker/")
            or p.endswith("/dockerfile")
            or p in ("docker-compose.yml", "docker-compose.yaml")
            for p in paths
        )
        # PEP 561 py.typed marker or stub .pyi files → type hints declared
        flags["has_type_hints"] = any(
            p == "py.typed" or p.endswith(".pyi")
            for p in paths
        )
        flags["has_examples"] = any(
            p.startswith(("examples/", "example/", "demos/", "demo/", "tutorials/"))
            or p == "examples"
            for p in paths
        )
        flags["has_benchmarks"] = any(
            p.startswith(("benchmarks/", "benchmark/", "perf/", "performance/"))
            or p == "benchmarks"
            for p in paths
        )

        return flags

    # ------------------------------------------------------------------
    # Per-repository metadata fetch
    # ------------------------------------------------------------------

    def fetch_metadata(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """Fetch all external metadata for one repository from the GitHub API.

        API calls made:
          GET /repos/{owner}/{repo}                 core info
          GET /repos/{owner}/{repo}/releases/latest release cadence
          GET /repos/{owner}/{repo}/contributors    contributor count
          GET /repos/{owner}/{repo}/topics          topic tags
          GET /repos/{owner}/{repo}/git/trees/HEAD  maturity flags

        Resilience:
          * If the core /repos endpoint fails, returns None (repo skipped).
          * All secondary endpoint failures produce safe defaults — the repo
            is never skipped due to a secondary failure.

        Args:
            repo_url: GitHub HTTPS clone URL, e.g. 'https://github.com/psf/requests.git'

        Returns:
            Flat dict of all collected metadata fields, or None only if the
            core repository info endpoint is unreachable.
        """
        owner, repo = self._owner_repo(repo_url)
        repo_name   = self._repo_name_from_url(repo_url)

        # ---- Core repository info (required) --------------------------------
        try:
            info = _gh_get(f"{_GH_API}/repos/{owner}/{repo}")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Exception fetching core info for %s: %s — skipping.", repo_name, exc
            )
            return None

        if not info:
            logger.warning(
                "Could not fetch core repo info for %s/%s — skipping.", owner, repo
            )
            return None

        created_at = _parse_dt(info.get("created_at"))
        pushed_at  = _parse_dt(info.get("pushed_at"))
        repo_age_days         = _days_since(created_at)
        days_since_last_commit = _days_since(pushed_at)

        # ---- Latest release (optional) --------------------------------------
        try:
            release_info = _gh_get(f"{_GH_API}/repos/{owner}/{repo}/releases/latest")
        except Exception:  # noqa: BLE001
            release_info = None

        latest_release_dt  = _parse_dt(release_info.get("published_at")) if release_info else None
        latest_release_tag = release_info.get("tag_name") if release_info else None
        days_since_release = _days_since(latest_release_dt)

        # ---- Contributors (first 100) ----------------------------------------
        try:
            contrib_data = _gh_get(
                f"{_GH_API}/repos/{owner}/{repo}/contributors",
                per_page=100, anon="false",
            )
            contributors_count = len(contrib_data) if isinstance(contrib_data, list) else 0
        except Exception:  # noqa: BLE001
            contributors_count = 0

        # ---- Topics ----------------------------------------------------------
        try:
            topics_resp = _gh_get(f"{_GH_API}/repos/{owner}/{repo}/topics")
            topics = ",".join(topics_resp.get("names", [])) if topics_resp else ""
        except Exception:  # noqa: BLE001
            topics = ""

        # ---- License ---------------------------------------------------------
        lic_block    = info.get("license") or {}
        license_name = (
            lic_block.get("spdx_id")
            or lic_block.get("name")
            or "NOASSERTION"
        )

        # ---- Maturity flags (never crashes the repo) -------------------------
        flags = self._detect_maturity_flags(owner, repo)

        return {
            # Identity
            "repository_name":       repo_name,
            "owner":                 owner,
            # Popularity
            "stars":                 info.get("stargazers_count", 0),
            "forks":                 info.get("forks_count", 0),
            "watchers":              info.get("watchers_count", 0),
            # Issues
            "open_issues":           info.get("open_issues_count", 0),
            "closed_issues":         None,          # not available without Search API
            # Community
            "contributors_count":    contributors_count,
            # Repository structure
            "default_branch":        info.get("default_branch", "main"),
            "license":               license_name,
            "primary_language":      info.get("language") or "Unknown",
            "topics":                topics,
            # Timestamps (ISO-8601 strings for CSV portability)
            "created_at":            info.get("created_at"),
            "updated_at":            info.get("updated_at"),
            "pushed_at":             info.get("pushed_at"),
            "latest_release_date":   release_info.get("published_at") if release_info else None,
            "latest_release_tag":    latest_release_tag,
            # Derived temporal signals
            "repo_age_days":            repo_age_days,
            "days_since_last_commit":   days_since_last_commit,
            "days_since_release":       days_since_release,
            # Maturity / quality flags
            **flags,
        }

    # ------------------------------------------------------------------
    # Batch metadata collection (used by repository_metadata_collector.py)
    # ------------------------------------------------------------------

    def fetch_all_metadata(
        self,
        inter_request_delay: float = 0.6,
        skip_names: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch metadata for repositories in the collection list.

        Args:
            inter_request_delay: Polite pause between repos (seconds).
            skip_names:          Set of repository_name strings already processed
                                 (used by the incremental resume logic in the
                                 metadata collector).  Matching repos are logged
                                 as skipped and excluded from the return list.

        Returns:
            List of metadata dicts — one per successfully fetched repository.
            Failed repos are logged as WARNING and excluded; collection continues.
        """
        skip_names   = skip_names or set()
        repo_urls    = self.get_repository_list()
        total        = len(repo_urls)
        records: List[Dict[str, Any]] = []

        for i, url in enumerate(repo_urls, start=1):
            repo_name = self._repo_name_from_url(url)
            if repo_name in skip_names:
                logger.info("[%d/%d] %s  (skipped — already processed)", i, total, repo_name)
                continue

            logger.info("[%d/%d] %s", i, total, repo_name)
            try:
                record = self.fetch_metadata(url)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[%d/%d] %s — unexpected exception: %s.  Continuing to next repo.",
                    i, total, repo_name, exc,
                )
                record = None

            if record:
                records.append(record)
                logger.info("Saved metadata for %s", repo_name)
            else:
                logger.warning(
                    "[%d/%d] %s — metadata fetch failed; repository skipped.",
                    i, total, repo_name,
                )

            time.sleep(inter_request_delay)

        return records

    # ------------------------------------------------------------------
    # Clone helpers (original functionality — unchanged)
    # ------------------------------------------------------------------

    def skip_existing_repositories(self, repo_url: str) -> bool:
        """Check if a repository is already cloned in the target directory.

        Args:
            repo_url: URL of the repository.

        Returns:
            True if the repository exists, False otherwise.
        """
        repo_name = repo_url.rstrip(".git").split("/")[-1]
        repo_path = self.target_dir / repo_name
        return repo_path.exists() and repo_path.is_dir()

    def clone_repository(self, repo_url: str) -> bool:
        """Clone a single repository.

        Args:
            repo_url: URL of the repository to clone.

        Returns:
            True if clone was successful or already exists, False otherwise.
        """
        repo_name = repo_url.rstrip(".git").split("/")[-1]
        repo_path = self.target_dir / repo_name

        if self.skip_existing_repositories(repo_url):
            logger.info(
                "Repository already exists. Skipping.", extra={"repository": repo_name}
            )
            return True

        logger.info(
            "Cloning repository", extra={"repository": repo_name, "url": repo_url}
        )
        try:
            self.target_dir.mkdir(parents=True, exist_ok=True)
            git.Repo.clone_from(repo_url, repo_path)
            logger.info(
                "Successfully cloned repository", extra={"repository": repo_name}
            )
            return True
        except Exception:
            logger.error(
                "Failed to clone repository",
                exc_info=True,
                extra={"repository": repo_name},
            )
            return False

    def clone_repositories(self) -> Dict[str, int]:
        """Clone all target repositories and return a summary.

        Returns:
            A dictionary with the counts of 'successful' and 'failed' clones.
        """
        repos   = self.get_repository_list()
        summary = {"successful": 0, "failed": 0}

        for repo_url in repos:
            success = self.clone_repository(repo_url)
            if success:
                summary["successful"] += 1
            else:
                summary["failed"] += 1

        logger.info("Cloning complete", extra={"summary": summary})
        return summary


# ---------------------------------------------------------------------------
# Stand-alone entry point (clone mode)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    collector = GitHubDatasetCollector()
    summary   = collector.clone_repositories()
    print(f"Summary: {summary}")
