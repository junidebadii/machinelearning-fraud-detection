"""Model training and evaluation module for fraud detection."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import joblib
import yaml
import logging
import argparse
from pathlib import Path

from .data import load_data, validate_schema, prepare_features, split_data
from .features import create_feature_pipeline

logger = logging.getLogger(__name__)

class FraudDetectionModel:
    """Fraud detection model wrapper."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_pipeline = None
        self.is_trained = False
        
    def create_pipeline(self) -> Pipeline:
        """Create the complete ML pipeline."""
        # Define feature types
        categorical_features = ['type']
        numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'balanceDiffOrig', 'balanceDiffDest']
        
        # Create feature preprocessing pipeline
        preprocessor = create_feature_pipeline(categorical_features, numerical_features)
        
        # Create model
        model_params = self.config.get('model', {}).get('params', {})
        classifier = LogisticRegression(**model_params)
        
        # Create complete pipeline
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', classifier)
        ])
        
        self.feature_pipeline = preprocessor
        self.model = pipeline
        logger.info("Created ML pipeline")
        return pipeline
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, float]:
        """Train the model."""
        if self.model is None:
            self.create_pipeline()
        
        # Train the pipeline
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Cross-validation scores
        cv_folds = self.config.get('cv', {}).get('folds', 3)
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring='roc_auc')
        
        logger.info(f"Model trained successfully. CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'cv_roc_auc_mean': cv_scores.mean(),
            'cv_roc_auc_std': cv_scores.std()
        }
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model on test data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': self.model.score(X_test, y_test),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred)
        }
        
        # Print detailed report
        logger.info("Model Evaluation Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"F1-Score: {metrics['f1']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_pred))
        
        return metrics
    
    def save_model(self, file_path: str):
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")
        
        joblib.dump(self.model, file_path)
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str):
        """Load a saved model."""
        self.model = joblib.load(file_path)
        self.is_trained = True
        logger.info(f"Model loaded from {file_path}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = self.model.predict(X)
        logger.info(f"Made predictions on {len(X)} samples")
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        probabilities = self.model.predict_proba(X)
        logger.info(f"Got prediction probabilities for {len(X)} samples")
        return probabilities

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--config', type=str, default='configs/train.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set random seed
    np.random.seed(config.get('random_seed', 42))
    
    # Load and prepare data
    data_path = config['data']['train_path']
    target_col = config['data']['target']
    
    logger.info(f"Loading data from {data_path}")
    df = load_data(data_path)
    
    # Validate schema
    if not validate_schema(df):
        raise ValueError("Data schema validation failed")
    
    # Prepare features
    df_processed = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(
        df_processed, target_col, 
        test_size=0.3, 
        random_state=config.get('random_seed', 42)
    )
    
    # Create and train model
    model = FraudDetectionModel(config)
    
    # Train model
    cv_results = model.train(X_train, y_train)
    
    # Evaluate model
    test_metrics = model.evaluate(X_test, y_test)
    
    # Save model
    model_path = 'artifacts/fraud_detection_pipeline.pkl'
    Path('artifacts').mkdir(exist_ok=True)
    model.save_model(model_path)
    
    logger.info("Training completed successfully!")
    logger.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
