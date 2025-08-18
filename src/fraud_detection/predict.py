"""Prediction module for fraud detection."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import joblib
import yaml
import logging
import argparse
from pathlib import Path

from .data import load_data, validate_schema, prepare_features
from .model import FraudDetectionModel

logger = logging.getLogger(__name__)

class FraudDetectionPredictor:
    """Fraud detection predictor for making predictions on new data."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model."""
        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Model loaded successfully from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {e}")
            raise
    
    def predict_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Make batch predictions."""
        try:
            predictions = self.model.predict(X)
            logger.info(f"Made predictions on {len(X)} samples")
            return predictions
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def predict_proba_batch(self, X: pd.DataFrame) -> np.ndarray:
        """Get batch prediction probabilities."""
        try:
            probabilities = self.model.predict_proba(X)
            logger.info(f"Got prediction probabilities for {len(X)} samples")
            return probabilities
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            raise
    
    def predict_single(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on a single transaction."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([transaction_data])
            
            # Prepare features
            df_processed = prepare_features(df)
            
            # Make prediction
            prediction = self.model.predict(df_processed)[0]
            probability = self.model.predict_proba(df_processed)[0][1]
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'is_fraud': bool(prediction == 1)
            }
            
            logger.info(f"Single prediction: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error making single prediction: {e}")
            raise

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_predictions(predictions: np.ndarray, output_path: str):
    """Save predictions to CSV file."""
    df_predictions = pd.DataFrame({
        'prediction': predictions,
        'is_fraud': predictions == 1
    })
    
    df_predictions.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")

def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(description='Make fraud detection predictions')
    parser.add_argument('--config', type=str, default='configs/infer.yaml', help='Path to config file')
    parser.add_argument('--input', type=str, help='Input data file path')
    parser.add_argument('--output', type=str, help='Output predictions file path')
    parser.add_argument('--model', type=str, help='Model file path')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    input_path = args.input or config['input_path']
    output_path = args.output or config['output_path']
    model_path = args.model or config['model_path']
    
    logger.info(f"Loading model from {model_path}")
    predictor = FraudDetectionPredictor(model_path)
    
    logger.info(f"Loading input data from {input_path}")
    df = load_data(input_path)
    
    # Validate schema
    if not validate_schema(df):
        raise ValueError("Data schema validation failed")
    
    # Prepare features
    df_processed = prepare_features(df)
    
    # Make predictions
    logger.info("Making predictions...")
    predictions = predictor.predict_batch(df_processed)
    
    # Save predictions
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_predictions(predictions, output_path)
    
    # Print summary
    fraud_count = np.sum(predictions == 1)
    total_count = len(predictions)
    fraud_rate = (fraud_count / total_count) * 100
    
    logger.info("Prediction Summary:")
    logger.info(f"Total transactions: {total_count}")
    logger.info(f"Fraudulent transactions: {fraud_count}")
    logger.info(f"Fraud rate: {fraud_rate:.2f}%")
    logger.info(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
