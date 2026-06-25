"""Configuration manager for the RepoGraph AI dataset pipeline.

Defines default constants, repository criteria, directory structure,
API configurations, and logging options.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Base Paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = WORKSPACE_ROOT / "data" / "dataset_pipeline"
RAW_REPOS_DIR = DATA_DIR / "raw_repos"
METADATA_DIR = DATA_DIR / "metadata"
METRICS_DIR = DATA_DIR / "metrics"
OUTPUT_DIR = DATA_DIR / "output"

# Ensure all directory structures exist
for directory in [DATA_DIR, RAW_REPOS_DIR, METADATA_DIR, METRICS_DIR, OUTPUT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# GitHub Search Criteria Configuration
GITHUB_API_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

TARGET_REPO_COUNT = 300
MIN_STARS = 100
MAX_REPOS_PER_SEARCH_QUERY = 100

# Search query patterns targeting popular active Python repositories
SEARCH_QUERIES: List[str] = [
    "language:python stars:>=100 size:>500 pushed:>=2025-01-01",
    "language:python topic:fastapi stars:>=50",
    "language:python topic:machine-learning stars:>=100",
    "language:python topic:rag stars:>=20",
]

# Collector Settings
CLONE_TIMEOUT_SECONDS = 300
MAX_CONCURRENT_CLONES = 5

# Rate Limiting & Retry Configuration
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0
RATE_LIMIT_BUFFER_SECONDS = 60

# Target Metrics Configuration
REQUIRED_FEATURES: List[str] = [
    "loc",
    "function_count",
    "class_count",
    "avg_function_length",
    "dependency_count",
    "fan_in",
    "fan_out",
    "module_count",
    "cyclomatic_complexity",
    "maintainability_index",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
]

def get_pipeline_config() -> Dict[str, Any]:
    """Retrieve runtime settings as a dictionary.

    TODO: Expand to support loading settings from a YAML or JSON configuration file.
    """
    return {
        "TARGET_REPO_COUNT": TARGET_REPO_COUNT,
        "MIN_STARS": MIN_STARS,
        "GITHUB_TOKEN_EXISTS": bool(GITHUB_TOKEN),
        "RAW_REPOS_DIR": str(RAW_REPOS_DIR),
        "OUTPUT_DIR": str(OUTPUT_DIR),
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("RepoGraph AI Pipeline Configuration loaded.")
    for key, val in get_pipeline_config().items():
        logger.info(f"{key}: {val}")
