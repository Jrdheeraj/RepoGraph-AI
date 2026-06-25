"""IQ Scorer for RepoGraph AI.

This module provides the IQScorer class, which trains a Random Forest model
and predicts a 0-100 IQ score for a repository based on structural and complexity metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np

try:
    from sklearn.ensemble import RandomForestRegressor
except ImportError:
    RandomForestRegressor = None

from ml.complexity_scorer import ComplexityScorer
from ml.metrics_extractor import MetricsExtractor

logger = logging.getLogger(__name__)


class IQScorer:
    """Scorer that calculates the Repository IQ Score using machine learning."""

    FEATURE_KEYS = [
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

    def __init__(self, model_path: Union[str, Path] = "iq_model.joblib") -> None:
        """Initialize the IQScorer.

        Args:
            model_path: Path to load/save the trained model.
        """
        self._model_path = Path(model_path)
        self._model: Optional[Any] = None
        if not RandomForestRegressor:
            logger.warning("scikit-learn is not installed. Model operations will fail.")

    def _get_model(self) -> Any:
        """Lazily load and return the trained model."""
        if self._model is None:
            self.load_model()
        return self._model

    def prepare_features(self, metrics: Dict[str, float]) -> np.ndarray:
        """Prepare and order the feature vector from a dictionary of metrics.

        Args:
            metrics: A dictionary containing the raw extracted metrics.

        Returns:
            A NumPy array containing the ordered features for model input.
        """
        logger.debug("Preparing features for model")
        feature_vector = []
        for key in self.FEATURE_KEYS:
            feature_vector.append(metrics.get(key, 0.0))
        return np.array(feature_vector).reshape(1, -1)

    def train(self, X_train: List[Dict[str, float]], y_train: List[float], **kwargs: Any) -> None:
        """Train the RandomForestRegressor model.

        Args:
            X_train: A list of metric dictionaries representing the training features.
            y_train: A list of float target values (IQ scores).
            **kwargs: Additional hyperparameters for the RandomForestRegressor.
        """
        logger.info("Training IQ scoring model", extra={"num_samples": len(X_train)})
        if not RandomForestRegressor:
            logger.error("scikit-learn is not available")
            return

        try:
            X_array = np.vstack([self.prepare_features(x) for x in X_train])
            y_array = np.array(y_train)

            self._model = RandomForestRegressor(**kwargs)
            self._model.fit(X_array, y_array)
            logger.info("Successfully trained model")
        except Exception as e:
            logger.error("Failed to train model", exc_info=True)

    def predict(self, metrics: Dict[str, float]) -> float:
        """Predict the repository IQ score based on metrics.

        Args:
            metrics: A dictionary containing the repository metrics.

        Returns:
            A float representing the repository IQ score (0.0 to 100.0).
        """
        logger.info("Predicting IQ score")
        try:
            model = self._get_model()
            if model is None:
                logger.error("No model available for prediction")
                return 0.0

            X = self.prepare_features(metrics)
            prediction = float(model.predict(X)[0])
            # Ensure the score is within 0-100 bounds
            score = max(0.0, min(100.0, prediction))
            logger.info("Successfully predicted IQ score", extra={"score": score})
            return score
        except Exception as e:
            logger.error("Failed to predict IQ score", exc_info=True)
            return 0.0

    def save_model(self) -> None:
        """Save the trained model to disk."""
        logger.info("Saving trained model", extra={"model_path": str(self._model_path)})
        try:
            if self._model is not None:
                self._model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self._model, self._model_path)
                logger.info("Successfully saved model")
            else:
                logger.error("No model to save")
        except Exception as e:
            logger.error("Failed to save model", exc_info=True)

    def load_model(self) -> None:
        """Load the trained model from disk."""
        logger.info("Loading trained model", extra={"model_path": str(self._model_path)})
        try:
            if self._model_path.exists():
                self._model = joblib.load(self._model_path)
                logger.info("Successfully loaded model")
            else:
                logger.warning("Model file not found", extra={"model_path": str(self._model_path)})
                self._model = None
        except Exception as e:
            logger.error("Failed to load model", exc_info=True)
            self._model = None

    def score_repository(self, repository_path: str | Path) -> float:
        """Extract metrics, predict, and return the repository IQ score.

        Args:
            repository_path: The root path of the repository to score.

        Returns:
            A float representing the final repository IQ score.
        """
        path = Path(repository_path)
        logger.info("Scoring repository IQ", extra={"repository_path": str(path)})
        try:
            # Extract basic metrics
            metrics_extractor = MetricsExtractor(repository_path=path)
            base_metrics = metrics_extractor.extract_metrics()

            # Extract complexity metrics
            complexity_scorer = ComplexityScorer()
            complexity_metrics = complexity_scorer.score_repository(repo_path=path)

            # Combine metrics
            combined_metrics = {**base_metrics, **complexity_metrics}

            # Predict score
            return self.predict(combined_metrics)
        except Exception as e:
            logger.error("Failed to score repository", exc_info=True, extra={"repository_path": str(path)})
            return 0.0
