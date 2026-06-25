"""Reusable utilities for the RepoGraph AI dataset pipeline.

Provides logging setup, JSON operations, rate-limiting handlers, HTTP header generators,
and command execution helpers.
"""

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def setup_pipeline_logging(module_name: str, level: int = logging.INFO) -> logging.Logger:
    """Configures and returns a module-specific logger.

    Args:
        module_name: Name of the active module.
        level: Logger level.
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    
    module_logger = logging.getLogger(module_name)
    module_logger.setLevel(level)
    if not module_logger.handlers:
        module_logger.addHandler(handler)
        
    return module_logger


def save_json(data: Any, file_path: Path) -> bool:
    """Saves data structures into a formatted JSON file.

    Args:
        data: Python dictionary or list to serialise.
        file_path: Output target path.
    """
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        logger.debug(f"Saved JSON data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save JSON data to {file_path}: {e}")
        return False


def load_json(file_path: Path) -> Optional[Any]:
    """Loads and returns JSON data from a file.

    Args:
        file_path: Source file path.
    """
    if not file_path.exists():
        logger.warning(f"JSON file does not exist: {file_path}")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON data from {file_path}: {e}")
        return None


def get_github_headers(token: Optional[str] = None) -> Dict[str, str]:
    """Generates request headers for interacting with GitHub API.

    Args:
        token: Optional GitHub Personal Access Token.
    """
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "RepoGraph-AI-Dataset-Pipeline"
    }
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def handle_rate_limit(reset_time: int) -> None:
    """Sleeps until the API rate limit resets.

    Args:
        reset_time: Unix timestamp of when the rate limit window resets.
    """
    current_time = int(time.time())
    sleep_duration = max(reset_time - current_time + 5, 10)
    logger.warning(f"Rate limit exceeded. Sleeping for {sleep_duration} seconds...")
    # TODO: Implement optional persistent delay queue or webhook triggers
    time.sleep(sleep_duration)


def run_command(command: str, cwd: Optional[Path] = None, timeout: int = 300) -> Optional[str]:
    """Executes a shell command synchronously and returns stdout.

    Args:
        command: The shell command sequence.
        cwd: Directory path in which to run command.
        timeout: Expiry threshold in seconds.
    """
    try:
        logger.debug(f"Running command: '{command}' in {cwd or 'default directory'}")
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
        return result.stdout.strip()
    except subprocess.TimeoutExpired:
        logger.error(f"Command execution timeout after {timeout} seconds: '{command}'")
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Command execution failure: '{command}' returned exit code {e.returncode}. stderr: {e.stderr}")
        return None


if __name__ == "__main__":
    log = setup_pipeline_logging("utils_test")
    log.info("Testing utility modules.")
    
    # Test JSON helper
    test_path = Path("test_data.json")
    save_json({"status": "ok", "message": "utils check"}, test_path)
    loaded = load_json(test_path)
    log.info(f"Loaded check: {loaded}")
    
    # Clean up test files
    if test_path.exists():
        test_path.unlink()
