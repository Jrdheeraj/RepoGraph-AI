"""Label Generator for the RepoGraph AI dataset pipeline.

Processes extracted metrics and metadata parameters to compute health labels,
maintainability risk scores, technical debt indices, and quality scores.
"""

import logging
from typing import Dict, Any
from pathlib import Path

try:
    from scripts.dataset_pipeline.config import OUTPUT_DIR
    from scripts.dataset_pipeline.utils import setup_pipeline_logging, save_json
except ImportError:
    from config import OUTPUT_DIR
    from utils import setup_pipeline_logging, save_json

logger = setup_pipeline_logging(__name__)


class LabelGenerator:
    """Class to construct synthetic and heuristic labels for model training."""

    def __init__(self, output_root: Path = OUTPUT_DIR):
        """Initializes the label generator.

        Args:
            output_root: Directory to write training label configurations.
        """
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _calculate_maintainability_risk(self, metrics: Dict[str, Any]) -> float:
        """Determines maintainability risk as a float from 0 to 100.

        Args:
            metrics: Extracted code metrics dictionary.
        """
        # TODO: Refine the heuristic risk formula using LOC, complexity, and maintainability index
        maintainability_index = metrics.get("maintainability_index", 100.0)
        cyclomatic_complexity = metrics.get("cyclomatic_complexity", 1.0)
        
        # Simple placeholder formula
        risk = (100.0 - maintainability_index) * 0.7 + (cyclomatic_complexity * 0.3)
        return float(max(0.0, min(100.0, risk)))

    def _calculate_iq_score(self, metrics: Dict[str, Any], metadata: Dict[str, Any]) -> float:
        """Determines codebase IQ score as a float from 0 to 100.

        Args:
            metrics: Code metrics.
            metadata: Repository metadata.
        """
        # TODO: Develop code quality score from release cadence, PR rates, and avg function length
        stars = metadata.get("stars", 0)
        contributors = metadata.get("total_contributors", 1)
        
        # Simple placeholder scoring using community size and basic stats
        popularity_modifier = min(stars / 1000.0 * 10, 15.0)
        contributor_modifier = min(contributors * 0.5, 10.0)
        
        base_quality = metrics.get("maintainability_index", 75.0)
        iq_score = base_quality + popularity_modifier + contributor_modifier
        return float(max(0.0, min(100.0, iq_score)))

    def _calculate_technical_debt(self, metrics: Dict[str, Any]) -> float:
        """Determines technical debt score as a float from 0 to 100.

        Args:
            metrics: Code metrics.
        """
        # TODO: Compute technical debt based on average function length and cyclomatic complexity
        avg_len = metrics.get("avg_function_length", 20.0)
        complexity = metrics.get("cyclomatic_complexity", 5.0)
        
        debt = (avg_len * 0.8) + (complexity * 1.5)
        return float(max(0.0, min(100.0, debt)))

    def _calculate_architecture_quality(self, metrics: Dict[str, Any]) -> float:
        """Determines architecture quality as a float from 0 to 100.

        Args:
            metrics: Code metrics.
        """
        # TODO: Evaluate coupling architecture using fan-in/fan-out metrics
        fan_in = metrics.get("fan_in", 0)
        fan_out = metrics.get("fan_out", 0)
        
        # Compute balanced coupling score
        coupling_ratio = (fan_in + 1) / (fan_out + 1)
        quality = 100.0 - (abs(1.0 - coupling_ratio) * 20.0)
        return float(max(0.0, min(100.0, quality)))

    def generate_labels(self, metrics: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, float]:
        """Calculates and aggregates all training labels for a repository.

        Args:
            metrics: Extracted code metrics features.
            metadata: Extracted repository metadata features.
        """
        logger.info("Generating training labels for repository...")
        
        risk = self._calculate_maintainability_risk(metrics)
        iq_score = self._calculate_iq_score(metrics, metadata)
        debt = self._calculate_technical_debt(metrics)
        arch_quality = self._calculate_architecture_quality(metrics)
        
        # Composite score matching prediction logic
        repograph_score = (
            0.35 * iq_score +
            0.25 * arch_quality +
            0.20 * (100.0 - risk) +
            0.20 * (100.0 - debt)
        )
        
        labels = {
            "iq_score": iq_score,
            "maintainability_risk": risk,
            "technical_debt_score": debt,
            "architecture_quality": arch_quality,
            "repograph_score": repograph_score
        }
        
        logger.info(f"Generated labels: {labels}")
        return labels

    def save_labels(self, repo_name: str, labels: Dict[str, float]) -> Path:
        """Saves calculated training labels to disk.

        Args:
            repo_name: Repository name.
            labels: Dict of computed labels.
        """
        output_file = self.output_root / f"{repo_name}__labels.json"
        save_json(labels, output_file)
        logger.info(f"Training labels saved to {output_file}")
        return output_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Executing Label Generator independently.")
    
    generator = LabelGenerator()
    dummy_metrics = {"maintainability_index": 82.0, "cyclomatic_complexity": 22, "avg_function_length": 15, "fan_in": 12, "fan_out": 6}
    dummy_metadata = {"stars": 500, "total_contributors": 12}
    
    labels = generator.generate_labels(dummy_metrics, dummy_metadata)
    output_path = generator.save_labels("test_repo", labels)
    
    # Clean up test files
    if output_path.exists():
        output_path.unlink()
