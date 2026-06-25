"""Dataset Validator for the RepoGraph AI dataset pipeline.

Validates the final aggregated dataset schema, value ranges, missing fields,
and check consistency against target count limits.
"""

import logging
from typing import Dict, Any, List
from pathlib import Path

try:
    from scripts.dataset_pipeline.config import REQUIRED_FEATURES
    from scripts.dataset_pipeline.utils import setup_pipeline_logging, load_json
except ImportError:
    from config import REQUIRED_FEATURES
    from utils import setup_pipeline_logging, load_json

logger = setup_pipeline_logging(__name__)


class DatasetValidator:
    """Validator class to verify clean, model-ready training datasets."""

    def __init__(self, target_count: int = 300):
        """Initializes the dataset validator.

        Args:
            target_count: Target number of records expected in the final dataset.
        """
        self.target_count = target_count

    def validate_record_schema(self, record: Dict[str, Any]) -> bool:
        """Verifies that a dataset record has all necessary columns.

        Args:
            record: Dictionary of features and labels for a single repository.
        """
        # TODO: Implement checks against required database features list and label columns.
        missing_features = [feat for feat in REQUIRED_FEATURES if feat not in record]
        if missing_features:
            logger.error(f"Record schema validation failed. Missing features: {missing_features}")
            return False
            
        required_labels = ["iq_score", "maintainability_risk", "technical_debt_score", "architecture_quality", "repograph_score"]
        missing_labels = [lbl for lbl in required_labels if lbl not in record]
        if missing_labels:
            logger.error(f"Record schema validation failed. Missing labels: {missing_labels}")
            return False
            
        return True

    def validate_value_ranges(self, record: Dict[str, Any]) -> bool:
        """Verifies value distributions and checks range constraints (e.g. scores between 0-100).

        Args:
            record: Dictionary of features and labels for a single repository.
        """
        # TODO: Implement outlier checks and numerical sanity boundaries.
        score_keys = ["iq_score", "maintainability_risk", "technical_debt_score", "architecture_quality", "repograph_score"]
        for key in score_keys:
            val = record.get(key, 0.0)
            if not (0.0 <= val <= 100.0):
                logger.error(f"Validation failed: '{key}' value {val} is outside valid score range [0, 100].")
                return False
                
        # Metric sanity checks
        if record.get("loc", 0) < 0:
            logger.error("Validation failed: 'loc' cannot be negative.")
            return False
            
        return True

    def check_dataset_completeness(self, dataset_records: List[Dict[str, Any]]) -> bool:
        """Evaluates whether the collected records meet targets.

        Args:
            dataset_records: Aggregated list of repository records.
        """
        record_count = len(dataset_records)
        logger.info(f"Checking dataset completeness. Records: {record_count}, Target: {self.target_count}")
        
        if record_count < self.target_count:
            logger.warning(f"Dataset has {record_count} records. It is under the target size of {self.target_count}.")
            # TODO: Raise alert or return False if strict target enforcement is configured.
            
        # Run validations on every record
        all_valid = True
        for idx, record in enumerate(dataset_records):
            if not (self.validate_record_schema(record) and self.validate_value_ranges(record)):
                logger.warning(f"Record index {idx} failed validation rules.")
                all_valid = False
                
        return all_valid


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Executing Dataset Validator independently.")
    
    validator = DatasetValidator(target_count=2)
    dummy_dataset = [
        {
            "loc": 1500, "function_count": 50, "class_count": 10, "avg_function_length": 30,
            "dependency_count": 5, "fan_in": 12, "fan_out": 8, "module_count": 3,
            "cyclomatic_complexity": 45, "maintainability_index": 75.5, "halstead_volume": 1200.0,
            "halstead_difficulty": 15.0, "halstead_effort": 18000.0,
            "iq_score": 80.0, "maintainability_risk": 20.0, "technical_debt_score": 25.0,
            "architecture_quality": 85.0, "repograph_score": 81.0
        },
        {
            "loc": -100,  # Invalid loc
            "function_count": 50, "class_count": 10, "avg_function_length": 30,
            "dependency_count": 5, "fan_in": 12, "fan_out": 8, "module_count": 3,
            "cyclomatic_complexity": 45, "maintainability_index": 75.5, "halstead_volume": 1200.0,
            "halstead_difficulty": 15.0, "halstead_effort": 18000.0,
            "iq_score": 120.0,  # Out of range
            "maintainability_risk": 20.0, "technical_debt_score": 25.0,
            "architecture_quality": 85.0, "repograph_score": 81.0
        }
    ]
    
    is_valid = validator.check_dataset_completeness(dummy_dataset)
    logger.info(f"Validation check completed. Result: {is_valid}")
