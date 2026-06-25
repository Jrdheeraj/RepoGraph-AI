"""GitHub Repository Discovery Engine for RepoGraph AI.

Queries the GitHub REST Search API across multiple technical categories, handles
token-based authentication, implements pagination, manages rate limits with
exponential backoff, caches searches to prevent redundant calls, deduplicates candidates,
and outputs a compiled CSV containing repository details.
"""

import os
import time
import logging
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import requests
from tqdm import tqdm

try:
    from scripts.dataset_pipeline.config import GITHUB_TOKEN
    from scripts.dataset_pipeline.utils import setup_pipeline_logging, save_json, load_json
except ImportError:
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
    def setup_pipeline_logging(name: str) -> logging.Logger:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        return logging.getLogger(name)
    def save_json(data: Any, file_path: Path) -> bool:
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            return True
        except Exception:
            return False
    def load_json(file_path: Path) -> Optional[Any]:
        if not file_path.exists():
            return None
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

logger = setup_pipeline_logging(__name__)

CACHE_FILE_PATH = Path("data/dataset_pipeline/search_cache.json")
OUTPUT_CSV_PATH = Path("data/datasets/repository_candidates.csv")
TARGET_COUNT = 300
MIN_STARS = 100
MIN_FORKS = 20
MIN_SIZE_KB = 100
MAX_REPOSITORY_AGE_MONTHS_WITHOUT_COMMITS = 24
CATEGORY_TARGET_MIN = 10
CATEGORY_TARGET_MAX = 15
QUERY_MAX_ITEMS = 30

# Structured search query patterns for high-signal technology domains.
SEARCH_CONFIG: Dict[str, List[str]] = {
    "FastAPI": [
        "fastapi language:python stars:>300",
        "topic:fastapi language:python stars:>100",
        "fastapi in:name language:python stars:>100",
    ],
    "Django": [
        "django language:python stars:>300",
        "topic:django language:python stars:>100",
        "django in:name language:python stars:>100",
    ],
    "Flask": [
        "flask language:python stars:>300",
        "topic:flask language:python stars:>100",
        "flask in:name language:python stars:>100",
    ],
    "React": [
        "react language:typescript stars:>300",
        "topic:react language:typescript stars:>100",
        "react in:name language:javascript stars:>100",
    ],
    "Next.js": [
        "nextjs language:typescript stars:>300",
        "topic:nextjs language:typescript stars:>100",
        "\"next.js\" language:typescript stars:>100",
    ],
    "Vue": [
        "vue language:typescript stars:>300",
        "topic:vue language:typescript stars:>100",
        "vue in:name language:javascript stars:>100",
    ],
    "Angular": [
        "angular language:typescript stars:>300",
        "topic:angular language:typescript stars:>100",
        "angular in:name language:typescript stars:>100",
    ],
    "TypeScript": [
        "typescript language:typescript stars:>500",
        "topic:typescript language:typescript stars:>300",
        "typescript in:name language:typescript stars:>100",
    ],
    "PyTorch": [
        "pytorch language:python stars:>300",
        "topic:pytorch language:python stars:>100",
        "pytorch in:name language:python stars:>100",
    ],
    "TensorFlow": [
        "tensorflow language:python stars:>300",
        "topic:tensorflow language:python stars:>100",
        "tensorflow in:name language:python stars:>100",
    ],
    "NumPy": [
        "numpy language:python stars:>300",
        "topic:numpy language:python stars:>100",
        "numpy in:name language:python stars:>100",
    ],
    "Pandas": [
        "pandas language:python stars:>300",
        "topic:pandas language:python stars:>100",
        "pandas in:name language:python stars:>100",
    ],
    "Scikit-learn": [
        "scikit-learn language:python stars:>300",
        "topic:scikit-learn language:python stars:>100",
        "sklearn language:python stars:>100",
    ],
    "LangChain": [
        "langchain language:python stars:>300",
        "topic:langchain language:python stars:>100",
        "langchain in:name language:python stars:>100",
    ],
    "LlamaIndex": [
        "llamaindex language:python stars:>300",
        "topic:llamaindex language:python stars:>100",
        "\"llama-index\" language:python stars:>100",
    ],
    "CrewAI": [
        "crewai language:python stars:>300",
        "topic:crewai language:python stars:>100",
        "crewai in:name language:python stars:>100",
    ],
    "AutoGen": [
        "autogen language:python stars:>300",
        "topic:autogen language:python stars:>100",
        "autogen in:name language:python stars:>100",
    ],
    "RAG": [
        "rag language:python stars:>300",
        "topic:rag language:python stars:>100",
        "\"retrieval augmented generation\" language:python stars:>100",
    ],
    "AI Agents": [
        "\"ai agents\" language:python stars:>300",
        "topic:ai-agents language:python stars:>100",
        "agentic language:python stars:>100",
    ],
    "MLOps": [
        "mlops language:python stars:>300",
        "topic:mlops language:python stars:>100",
        "\"machine learning operations\" language:python stars:>100",
    ],
    "Docker": [
        "docker language:go stars:>300",
        "topic:docker stars:>100",
        "docker in:name stars:>100",
    ],
    "Kubernetes": [
        "kubernetes language:go stars:>300",
        "topic:kubernetes stars:>100",
        "kubernetes in:name stars:>100",
    ],
    "Terraform": [
        "terraform language:go stars:>300",
        "topic:terraform stars:>100",
        "terraform in:name stars:>100",
    ],
    "Airflow": [
        "airflow language:python stars:>300",
        "topic:airflow language:python stars:>100",
        "airflow in:name language:python stars:>100",
    ],
    "Ray": [
        "ray language:python stars:>300",
        "topic:ray language:python stars:>100",
        "ray in:name language:python stars:>100",
    ],
    "Redis": [
        "redis language:c stars:>300",
        "topic:redis stars:>100",
        "redis in:name stars:>100",
    ],
    "PostgreSQL": [
        "postgresql language:c stars:>300",
        "topic:postgresql stars:>100",
        "postgres in:name stars:>100",
    ],
    "Kafka": [
        "kafka language:java stars:>300",
        "topic:kafka stars:>100",
        "kafka in:name stars:>100",
    ],
    "GraphQL": [
        "graphql language:typescript stars:>300",
        "topic:graphql stars:>100",
        "graphql in:name stars:>100",
    ],
    "Cyber Security": [
        "cybersecurity language:python stars:>300",
        "topic:cybersecurity stars:>100",
        "\"security tools\" language:python stars:>100",
    ],
}


class GitHubDiscoveryEngine:
    """Production-grade discovery engine querying GitHub REST Search API."""

    def __init__(self, token: Optional[str] = None):
        """Initializes client session with authentication headers.

        Args:
            token: GitHub Personal Access Token.
        """
        self.token = token or GITHUB_TOKEN
        self.headers = {
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "RepoGraph-AI-Discovery-Engine"
        }
        if self.token:
            self.headers["Authorization"] = f"token {self.token}"
            logger.info("GitHub Discovery Engine initialized with token authentication.")
        else:
            logger.warning("No GITHUB_TOKEN environment variable found. API requests will hit strict rate limits.")

    def search_query(self, query: str, max_items: int = 50) -> List[Dict[str, Any]]:
        """Queries the repository search API supporting pagination and rate limit backoff.

        Args:
            query: Query string to match repositories.
            max_items: Maximum items to collect for this query pattern.
        """
        results: List[Dict[str, Any]] = []
        page = 1
        per_page = min(max_items, 100)
        
        while len(results) < max_items:
            url = f"https://api.github.com/search/repositories?q={query}&page={page}&per_page={per_page}"
            retries = 0
            max_retries = 5
            backoff = 2.0
            
            response = None
            while retries < max_retries:
                try:
                    logger.debug(f"Requesting URL: {url}")
                    response = requests.get(url, headers=self.headers, timeout=15)
                    
                    if response.status_code == 200:
                        break
                    
                    # Handle API rate limits (HTTP 403 or 429)
                    if response.status_code in (403, 429):
                        reset_header = response.headers.get("X-RateLimit-Reset")
                        sleep_time = 60
                        if reset_header:
                            try:
                                reset_epoch = int(reset_header)
                                sleep_time = max(reset_epoch - int(time.time()) + 2, 2)
                            except ValueError:
                                pass
                        logger.warning(f"Rate limited (status {response.status_code}). Sleeping for {sleep_time} seconds...")
                        time.sleep(sleep_time)
                        retries += 1
                        continue
                        
                    logger.warning(f"Received status code {response.status_code} for search. Retrying...")
                    time.sleep(backoff ** retries)
                    retries += 1
                except requests.RequestException as e:
                    logger.warning(f"Connection error: {e}. Retrying...")
                    time.sleep(backoff ** retries)
                    retries += 1
            
            if not response or response.status_code != 200:
                logger.error(f"Failed to fetch page {page} for query '{query}' after {max_retries} retries.")
                break
                
            data = response.json()
            items = data.get("items", [])
            if not items:
                break
                
            results.extend(items)
            page += 1
            
            # Rest standard Search API rate spacing (max 30 requests per min for authenticated)
            time.sleep(2)
            
        return results[:max_items]


def load_cache() -> Dict[str, List[Dict[str, Any]]]:
    """Loads query results from a local JSON cache file."""
    cache = load_json(CACHE_FILE_PATH)
    return cache if isinstance(cache, dict) else {}


def save_cache(cache: Dict[str, List[Dict[str, Any]]]) -> None:
    """Saves compiled query results locally to a JSON cache file."""
    save_json(cache, CACHE_FILE_PATH)


def parse_github_datetime(value: Optional[str]) -> Optional[datetime]:
    """Parses GitHub API timestamps into timezone-aware datetime objects."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def repository_dedupe_key(repo: Dict[str, Any]) -> Tuple[str, str]:
    """Returns a stable deduplication key based on owner and repository name."""
    owner = repo.get("owner", {}).get("login", "")
    name = repo.get("name", "")
    return owner.lower(), name.lower()


def months_since(timestamp: Optional[str]) -> Optional[float]:
    """Returns approximate months elapsed since a GitHub timestamp."""
    parsed_timestamp = parse_github_datetime(timestamp)
    if not parsed_timestamp:
        return None
    elapsed_days = (datetime.now(timezone.utc) - parsed_timestamp).days
    return max(elapsed_days / 30.4375, 0)


def repository_filter_reason(repo: Dict[str, Any]) -> Optional[str]:
    """Returns the exclusion reason when a repository fails quality filters."""
    if repo.get("archived", False):
        return "archived"
    if repo.get("disabled", False):
        return "disabled"
    if repo.get("fork", False):
        return "fork"
    if repo.get("is_template", False):
        return "template"
    if int(repo.get("stargazers_count") or 0) < MIN_STARS:
        return "stars"
    if int(repo.get("forks_count") or 0) < MIN_FORKS:
        return "forks"
    if int(repo.get("size") or 0) < MIN_SIZE_KB:
        return "size"

    months_since_push = months_since(repo.get("pushed_at"))
    if months_since_push is None or months_since_push > MAX_REPOSITORY_AGE_MONTHS_WITHOUT_COMMITS:
        return "stale"

    return None


def compute_repository_quality(repo: Dict[str, Any]) -> float:
    """Computes a bounded repository quality score from GitHub metadata.

    The score emphasizes community adoption, active maintenance, metadata
    completeness, and repository health while keeping the output in [0, 100].
    """
    stars = int(repo.get("stargazers_count") or 0)
    forks = int(repo.get("forks_count") or 0)
    open_issues = int(repo.get("open_issues_count") or 0)
    topics = repo.get("topics") or []
    watchers = int(repo.get("watchers_count") or 0)
    pushed_months = months_since(repo.get("pushed_at"))
    created_months = months_since(repo.get("created_at"))

    stars_score = min(stars / 5000, 1.0) * 22
    forks_score = min(forks / 1000, 1.0) * 14

    if pushed_months is None:
        activity_score = 0.0
    elif pushed_months <= 1:
        activity_score = 18.0
    elif pushed_months <= 6:
        activity_score = 15.0
    elif pushed_months <= 12:
        activity_score = 10.0
    elif pushed_months <= 24:
        activity_score = 5.0
    else:
        activity_score = 0.0

    topics_score = min(len(topics), 8) / 8 * 8
    issue_ratio = open_issues / max(stars, 1)
    issues_score = max(0.0, 10.0 - min(issue_ratio * 250, 10.0))

    if created_months is None:
        age_score = 0.0
    elif created_months < 3:
        age_score = 2.0
    elif created_months <= 84:
        age_score = 8.0
    else:
        age_score = 6.0

    license_score = 6.0 if repo.get("license") else 0.0
    homepage_score = 4.0 if repo.get("homepage") else 0.0
    wiki_score = 4.0 if repo.get("has_wiki") else 0.0
    default_branch = str(repo.get("default_branch") or "").lower()
    branch_score = 3.0 if default_branch in {"main", "master"} else 1.5 if default_branch else 0.0
    watchers_score = min(watchers / 5000, 1.0) * 3

    score = (
        stars_score
        + forks_score
        + activity_score
        + topics_score
        + issues_score
        + age_score
        + license_score
        + homepage_score
        + wiki_score
        + branch_score
        + watchers_score
    )
    return round(max(0.0, min(score, 100.0)), 2)


def format_license(repo: Dict[str, Any]) -> str:
    """Returns a compact license identifier from GitHub repository metadata."""
    license_info = repo.get("license") or {}
    if not isinstance(license_info, dict):
        return ""
    return license_info.get("spdx_id") or license_info.get("key") or license_info.get("name") or ""


def discover_repositories() -> None:
    """Orchestrates candidate collection across all categories and saves to CSV."""
    logger.info("Starting GitHub Repository Discovery Pipeline...")
    
    engine = GitHubDiscoveryEngine()
    cache = load_cache()
    all_repos: Dict[Tuple[str, str], Dict[str, Any]] = {}
    discovered_count = 0
    removed_by_filters = 0
    removed_as_duplicates = 0
    per_category_limit = min(
        CATEGORY_TARGET_MAX,
        max(CATEGORY_TARGET_MIN, TARGET_COUNT // max(len(SEARCH_CONFIG), 1)),
    )
    
    # Iterate through structured categories, extract items, filter, score, and balance.
    for category, queries in tqdm(SEARCH_CONFIG.items(), desc="Processing Categories"):
        category_repos: Dict[Tuple[str, str], Dict[str, Any]] = {}
        category_discovered = 0
        category_filtered = 0
        category_duplicates = 0

        for query in queries:
            if query in cache:
                logger.info(f"Loaded cached repositories for category '{category}' query: {query}")
                items = cache[query]
            else:
                logger.info(f"Executing live search for category '{category}' query: {query}")
                items = engine.search_query(query, max_items=QUERY_MAX_ITEMS)
                cache[query] = items
                save_cache(cache)

            category_discovered += len(items)
            discovered_count += len(items)

            for item in items:
                dedupe_key = repository_dedupe_key(item)
                if not all(dedupe_key):
                    category_filtered += 1
                    removed_by_filters += 1
                    continue

                if repository_filter_reason(item):
                    category_filtered += 1
                    removed_by_filters += 1
                    continue

                item["repository_quality_score"] = compute_repository_quality(item)
                item["category"] = category

                existing_category_repo = category_repos.get(dedupe_key)
                if existing_category_repo:
                    category_duplicates += 1
                    removed_as_duplicates += 1
                    if item["repository_quality_score"] <= existing_category_repo["repository_quality_score"]:
                        continue

                category_repos[dedupe_key] = item

        balanced_category_repos = sorted(
            category_repos.values(),
            key=lambda repo: repo.get("repository_quality_score", 0),
            reverse=True,
        )[:per_category_limit]

        for repo in balanced_category_repos:
            dedupe_key = repository_dedupe_key(repo)
            existing_repo = all_repos.get(dedupe_key)
            if existing_repo:
                removed_as_duplicates += 1
                if repo["repository_quality_score"] <= existing_repo["repository_quality_score"]:
                    continue
            all_repos[dedupe_key] = repo

        logger.info(
            "Category '%s': discovered=%s, filtered=%s, duplicates=%s, kept=%s",
            category,
            category_discovered,
            category_filtered,
            category_duplicates,
            len(balanced_category_repos),
        )

    if len(all_repos) > TARGET_COUNT:
        all_repos = dict(
            sorted(
                all_repos.items(),
                key=lambda item: item[1].get("repository_quality_score", 0),
                reverse=True,
            )[:TARGET_COUNT]
        )

    logger.info(f"Discovered {discovered_count} repository candidates before quality filters.")
    logger.info(f"Removed {removed_by_filters} repositories by quality filters.")
    logger.info(f"Removed {removed_as_duplicates} repositories as duplicates.")
    logger.info(f"Kept {len(all_repos)} unique repository candidates after balancing.")
    
    # Process and transform fields
    processed_records = []
    for repo in all_repos.values():
        owner = repo.get("owner", {}).get("login", "")
        repository_name = repo.get("name", "")
        html_url = repo.get("html_url", "")
        language = repo.get("language", "")
        stars = repo.get("stargazers_count", 0)
        forks = repo.get("forks_count", 0)
        topics = ",".join(repo.get("topics", []))
        default_branch = repo.get("default_branch", "")
        created_at = repo.get("created_at", "")
        updated_at = repo.get("updated_at", "")
        pushed_at = repo.get("pushed_at", "")
        
        processed_records.append({
            "owner": owner,
            "repository_name": repository_name,
            "html_url": html_url,
            "language": language,
            "stars": stars,
            "forks": forks,
            "repository_quality_score": repo.get("repository_quality_score", 0),
            "archived": repo.get("archived", False),
            "disabled": repo.get("disabled", False),
            "fork": repo.get("fork", False),
            "license": format_license(repo),
            "homepage": repo.get("homepage", ""),
            "watchers": repo.get("watchers_count", 0),
            "subscribers": repo.get("subscribers_count", 0),
            "open_issues": repo.get("open_issues_count", 0),
            "default_branch": default_branch,
            "size": repo.get("size", 0),
            "description": repo.get("description", ""),
            "category": repo.get("category", ""),
            "topics": topics,
            "created_at": created_at,
            "updated_at": updated_at,
            "pushed_at": pushed_at
        })
        
    # Convert to DataFrame
    df = pd.DataFrame(processed_records)
    
    # Ensure directory exists and write CSV output
    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8")

    average_quality_score = (
        round(float(df["repository_quality_score"].mean()), 2)
        if not df.empty and "repository_quality_score" in df
        else 0.0
    )
    final_category_distribution = (
        df["category"].value_counts().sort_index().to_dict()
        if not df.empty and "category" in df
        else {}
    )
    
    logger.info(f"Average repository quality score: {average_quality_score}")
    logger.info(f"Category distribution: {final_category_distribution}")
    logger.info(f"Successfully saved {len(df)} candidate repositories to {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    discover_repositories()
