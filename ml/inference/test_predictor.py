import json
import logging
from predictor import predict_repo_health

# Configure basic logging for the test script
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """
    Test script to validate the functionality of the RepoGraph Inference Engine.
    """
    # 1. Create a sample repository metrics dictionary with all 13 required features
    sample_features = {
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
    
    print("=== RepoGraph Inference Engine Test ===")
    print("\n[Input Features]")
    print(json.dumps(sample_features, indent=4))
    
    # 2. Run prediction
    try:
        logger.info("Executing prediction...")
        results = predict_repo_health(sample_features)
        
        # 3. Print formatted JSON output
        print("\n[Prediction Results]")
        print(json.dumps(results, indent=4))
        
    except Exception as e:
        logger.error(f"Failed during prediction test: {e}")
        print("\n[Error]")
        print(f"Test failed: {e}")

if __name__ == "__main__":
    main()
