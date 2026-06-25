"""Anomaly Detector for RepoGraph AI.

This module provides the AnomalyDetector class, which trains an Isolation Forest model
and predicts whether a repository is an architectural anomaly based on structural and complexity metrics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np

try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    IsolationForest = None

from ml.complexity_scorer import ComplexityScorer
from ml.metrics_extractor import MetricsExtractor

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detector that flags architectural anomalies using machine learning.
    
    Implementation meets the strict requirements using IsolationForest.
    """

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

    def __init__(self, model_path: Union[str, Path] = "anomaly_model.joblib") -> None:
        """Initialize the AnomalyDetector.

        Args:
            model_path: Path to load/save the trained model.
        """
        self._model_path = Path(model_path)
        self._model: Optional[Any] = None
        if not IsolationForest:
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
        logger.debug("Preparing features for anomaly detection model")
        feature_vector = []
        for key in self.FEATURE_KEYS:
            feature_vector.append(metrics.get(key, 0.0))
        return np.array(feature_vector).reshape(1, -1)

    def train(self, X_train: List[Dict[str, float]], **kwargs: Any) -> None:
        """Train the IsolationForest model.

        Args:
            X_train: A list of metric dictionaries representing the training features.
            **kwargs: Additional hyperparameters for the IsolationForest.
        """
        logger.info("Training anomaly detection model", extra={"num_samples": len(X_train)})
        if not IsolationForest:
            logger.error("scikit-learn is not available")
            return

        try:
            if not X_train:
                logger.error("No training data provided")
                return

            X_array = np.vstack([self.prepare_features(x) for x in X_train])

            self._model = IsolationForest(**kwargs)
            self._model.fit(X_array)
            logger.info("Successfully trained anomaly detection model")
        except Exception as e:
            logger.error("Failed to train anomaly detection model", exc_info=True)

    def predict(self, metrics: Dict[str, float]) -> Dict[str, Union[bool, float]]:
        """Predict whether the given metrics represent an anomaly.

        Args:
            metrics: A dictionary containing the repository metrics.

        Returns:
            A dictionary with 'is_anomaly' (bool) and 'anomaly_score' (float).
        """
        logger.info("Predicting anomaly")
        default_result = {"is_anomaly": False, "anomaly_score": 0.0}

        try:
            model = self._get_model()
            if model is None:
                logger.error("No model available for prediction")
                return default_result

            X = self.prepare_features(metrics)
            
            # IsolationForest predict returns -1 for outliers and 1 for inliers
            prediction = model.predict(X)[0]
            is_anomaly = bool(prediction == -1)
            
            score = float(model.decision_function(X)[0])
            
            result = {
                "is_anomaly": is_anomaly,
                "anomaly_score": score
            }
            logger.info("Successfully predicted anomaly", extra=result)
            return result
        except Exception as e:
            logger.error("Failed to predict anomaly", exc_info=True)
            return default_result

    def save_model(self) -> None:
        """Save the trained model to disk."""
        logger.info("Saving trained anomaly model", extra={"model_path": str(self._model_path)})
        try:
            if self._model is not None:
                self._model_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(self._model, self._model_path)
                logger.info("Successfully saved anomaly model")
            else:
                logger.error("No model to save")
        except Exception as e:
            logger.error("Failed to save anomaly model", exc_info=True)

    def load_model(self) -> None:
        """Load the trained model from disk."""
        logger.info("Loading trained anomaly model", extra={"model_path": str(self._model_path)})
        try:
            if self._model_path.exists():
                self._model = joblib.load(self._model_path)
                logger.info("Successfully loaded anomaly model")
            else:
                logger.warning("Anomaly model file not found", extra={"model_path": str(self._model_path)})
                self._model = None
        except Exception as e:
            logger.error("Failed to load anomaly model", exc_info=True)
            self._model = None

    def detect_repository(self, repository_path: str | Path) -> Dict[str, Union[bool, float]]:
        """Extract metrics, predict, and return the repository anomaly score.

        Args:
            repository_path: The root path of the repository to analyze.

        Returns:
            A dictionary with 'is_anomaly' (bool) and 'anomaly_score' (float).
        """
        path = Path(repository_path)
        logger.info("Detecting anomalies for repository", extra={"repository_path": str(path)})
        
        default_result = {"is_anomaly": False, "anomaly_score": 0.0}
        
        try:
            # Extract basic metrics
            metrics_extractor = MetricsExtractor(repository_path=path)
            base_metrics = metrics_extractor.extract_metrics()

            # Extract complexity metrics
            complexity_scorer = ComplexityScorer()
            complexity_metrics = complexity_scorer.score_repository(repo_path=path)

            # Combine metrics
            combined_metrics = {**base_metrics, **complexity_metrics}

            # Predict anomaly
            return self.predict(combined_metrics)
        except Exception as e:
            logger.error("Failed to detect repository anomaly", exc_info=True, extra={"repository_path": str(path)})
            return default_result
