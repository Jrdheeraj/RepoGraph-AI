import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
FEATURES = [
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
    "halstead_effort"
]

# Resolve models directory relative to this file
# This points to ml/models/ regardless of where the script is executed from
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Global model variables
_iq_score_model = None
_risk_model = None
_debt_model = None
_architecture_model = None


def _load_models() -> None:
    """
    Loads machine learning models into global variables during module initialization.
    """
    global _iq_score_model, _risk_model, _debt_model, _architecture_model
    
    try:
        logger.info(f"Loading models from {MODELS_DIR}...")
        
        # Load each model using joblib
        _iq_score_model = joblib.load(MODELS_DIR / "iq_score_model.pkl")
        _risk_model = joblib.load(MODELS_DIR / "risk_model.pkl")
        _debt_model = joblib.load(MODELS_DIR / "debt_model.pkl")
        _architecture_model = joblib.load(MODELS_DIR / "architecture_model.pkl")
        
        logger.info("Successfully loaded all models.")
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}. Please ensure models are generated.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")


# Execute model loading on module import
_load_models()


def _clamp(value: float, min_val: float = 0.0, max_val: float = 100.0) -> float:
    """
    Clamps a value between min_val and max_val.
    """
    return max(min_val, min(value, max_val))


def predict_repo_health(features_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Generates health predictions for a repository based on input features.

    Args:
        features_dict (Dict[str, Any]): Dictionary containing repository metrics.

    Returns:
        Dict[str, float]: Dictionary containing predicted scores.
        
    Raises:
        ValueError: If required features are missing or models are not loaded.
        Exception: For any other inference-related errors.
    """
    # 1. Validate models are loaded
    if any(m is None for m in [_iq_score_model, _risk_model, _debt_model, _architecture_model]):
        logger.error("Models are not loaded properly.")
        raise ValueError("Inference engine is not initialized. Models are missing or failed to load.")

    # 2. Validate input features
    missing_features = [f for f in FEATURES if f not in features_dict]
    if missing_features:
        logger.error(f"Missing required features: {missing_features}")
        raise ValueError(f"Input is missing required features: {missing_features}")

    try:
        # 3. Convert to DataFrame
        # Pass index=[0] because we are predicting a single instance
        df = pd.DataFrame([features_dict])

        # 4. Reorder columns to match FEATURES exactly
        df = df[FEATURES]

        # 5. Generate predictions
        # Extract the first element [0] from the prediction array
        iq_score = float(_iq_score_model.predict(df)[0])
        maintainability_risk = float(_risk_model.predict(df)[0])
        technical_debt_score = float(_debt_model.predict(df)[0])
        architecture_quality = float(_architecture_model.predict(df)[0])

        # 6. Compute RepoGraph score
        repograph_score = (
            0.35 * iq_score +
            0.25 * architecture_quality +
            0.20 * (100 - maintainability_risk) +
            0.20 * (100 - technical_debt_score)
        )

        # 7. Clamp all outputs between 0 and 100
        results = {
            "iq_score": _clamp(iq_score),
            "maintainability_risk": _clamp(maintainability_risk),
            "technical_debt_score": _clamp(technical_debt_score),
            "architecture_quality": _clamp(architecture_quality),
            "repograph_score": _clamp(repograph_score)
        }
        
        logger.info("Successfully generated health predictions.")
        return results

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
