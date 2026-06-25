import logging
from typing import Dict, Any
import sys
import os

# Add the project root to sys.path so we can import modules properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from ml.inference.predictor import predict_repo_health

# Assuming standard entry points based on the project structure
try:
    from parser.repo_scanner import scan_repo # type: ignore
except ImportError:
    scan_repo = None

try:
    from metrics.extractor import extract_metrics # type: ignore
except ImportError:
    extract_metrics = None

logger = logging.getLogger(__name__)

def analyze_repository(github_url: str) -> Dict[str, Any]:
    """
    Analyzes a GitHub repository and returns health scores.

    Args:
        github_url (str): URL of the GitHub repository.

    Returns:
        Dict[str, Any]: A dictionary containing the repository name and computed scores.
        
    Raises:
        ValueError: If URL is invalid or parsing fails.
        Exception: If any step in the pipeline fails.
    """
    logger.info(f"Starting analysis for repository: {github_url}")
    
    # Extract simple repository name from URL for display purposes
    repository_name = github_url.rstrip('/').split('/')[-1]

    try:
        # Step 1: Parse the repository
        logger.info("Step 1: Parsing repository...")
        if scan_repo:
            parsed_data = scan_repo(github_url)
        else:
            logger.warning("Parser module not found or incomplete. Using mock parsed data.")
            parsed_data = {"repo_url": github_url}

        # Step 2: Extract metrics
        logger.info("Step 2: Extracting metrics...")
        if extract_metrics:
            features_dict = extract_metrics(parsed_data)
        else:
            logger.warning("Metrics extractor not found or incomplete. Using mock features.")
            features_dict = {
                "loc": 1500,
                "function_count": 50,
                "class_count": 10,
                "avg_function_length": 30,
                "dependency_count": 5,
                "fan_in": 12,
                "fan_out": 8,
                "module_count": 3,
                "cyclomatic_complexity": 45,
                "maintainability_index": 75.5,
                "halstead_volume": 1200.0,
                "halstead_difficulty": 15.0,
                "halstead_effort": 18000.0
            }

        # Step 3: Predict health using ML Models
        logger.info("Step 3: Predicting repository health...")
        prediction_results = predict_repo_health(features_dict)

        # Step 4: Construct and return result dictionary
        result = {
            "repository": repository_name,
            **prediction_results
        }
        
        logger.info(f"Successfully completed analysis for {repository_name}")
        return result

    except ValueError as ve:
        logger.error(f"Validation error during repository analysis: {str(ve)}")
        raise
    except Exception as e:
        logger.error(f"Error during repository analysis pipeline: {str(e)}")
        raise
