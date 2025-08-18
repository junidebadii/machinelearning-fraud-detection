"""Smoke tests for the fraud detection model."""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection.model import FraudDetectionModel
from fraud_detection.data import prepare_features

class TestModelSmoke:
    """Smoke tests for model functionality."""
    
    def setup_method(self):
        """Setup test data and model."""
        # Create minimal test data
        self.test_data = pd.DataFrame({
            'step': [1, 2, 3, 4, 5],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEPOSIT', 'PAYMENT'],
            'amount': [100.0, 200.0, 300.0, 400.0, 500.0],
            'nameOrig': ['C123', 'C456', 'C789', 'C012', 'C345'],
            'oldbalanceOrg': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            'newbalanceOrig': [900.0, 1800.0, 2700.0, 3600.0, 4500.0],
            'nameDest': ['M123', 'M456', 'M789', 'M012', 'M345'],
            'oldbalanceDest': [0.0, 0.0, 0.0, 0.0, 0.0],
            'newbalanceDest': [100.0, 200.0, 300.0, 400.0, 500.0],
            'isFraud': [0, 0, 1, 0, 0],
            'isFlaggedFraud': [0, 0, 0, 0, 0]
        })
        
        # Model configuration
        self.config = {
            'model': {
                'type': 'sklearn.LogisticRegression',
                'params': {
                    'max_iter': 100,
                    'class_weight': 'balanced'
                }
            },
            'cv': {
                'folds': 2
            }
        }
        
        self.model = FraudDetectionModel(self.config)
    
    def test_model_creation(self):
        """Test that model can be created."""
        assert self.model is not None
        assert self.model.config == self.config
        assert self.model.is_trained == False
    
    def test_pipeline_creation(self):
        """Test that ML pipeline can be created."""
        pipeline = self.model.create_pipeline()
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'predict')
    
    def test_end_to_end_training(self):
        """Test complete training pipeline."""
        # Prepare features
        df_processed = prepare_features(self.test_data)
        
        # Split data
        X = df_processed.drop(columns=['isFraud'])
        y = df_processed['isFraud']
        
        # Train model
        cv_results = self.model.train(X, y)
        
        # Check results
        assert self.model.is_trained == True
        assert 'cv_roc_auc_mean' in cv_results
        assert 'cv_roc_auc_std' in cv_results
        assert cv_results['cv_roc_auc_mean'] >= 0.0
        assert cv_results['cv_roc_auc_mean'] <= 1.0
    
    def test_model_evaluation(self):
        """Test model evaluation functionality."""
        # Prepare and train
        df_processed = prepare_features(self.test_data)
        X = df_processed.drop(columns=['isFraud'])
        y = df_processed['isFraud']
        
        self.model.train(X, y)
        
        # Evaluate
        metrics = self.model.evaluate(X, y)
        
        # Check metrics
        required_metrics = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
        for metric in required_metrics:
            assert metric in metrics
            assert metrics[metric] >= 0.0
            assert metrics[metric] <= 1.0
    
    def test_model_prediction(self):
        """Test model prediction functionality."""
        # Prepare and train
        df_processed = prepare_features(self.test_data)
        X = df_processed.drop(columns=['isFraud'])
        y = df_processed['isFraud']
        
        self.model.train(X, y)
        
        # Make predictions
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Check predictions
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Check probabilities
        assert probabilities.shape == (len(X), 2)  # Binary classification
        assert all(prob >= 0.0 and prob <= 1.0 for prob in probabilities.flatten())
        assert all(np.isclose(prob.sum(axis=1), 1.0))  # Probabilities sum to 1
    
    def test_model_save_load(self, tmp_path):
        """Test model saving and loading."""
        # Prepare and train
        df_processed = prepare_features(self.test_data)
        X = df_processed.drop(columns=['isFraud'])
        y = df_processed['isFraud']
        
        self.model.train(X, y)
        
        # Save model
        save_path = tmp_path / "model.pkl"
        self.model.save_model(str(save_path))
        assert save_path.exists()
        
        # Load model
        new_model = FraudDetectionModel(self.config)
        new_model.load_model(str(save_path))
        
        # Test that loaded model works
        assert new_model.is_trained == True
        predictions = new_model.predict(X)
        assert len(predictions) == len(X)
    
    def test_model_without_training(self):
        """Test that untrained model raises appropriate errors."""
        # Test prediction without training
        df_processed = prepare_features(self.test_data)
        X = df_processed.drop(columns=['isFraud'])
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            self.model.predict(X)
        
        with pytest.raises(ValueError, match="Model must be trained before making predictions"):
            self.model.predict_proba(X)
        
        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            self.model.evaluate(X, pd.Series([0, 0, 0, 0, 0]))
        
        with pytest.raises(ValueError, match="Cannot save untrained model"):
            self.model.save_model("dummy.pkl")
