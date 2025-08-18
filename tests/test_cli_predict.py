"""Tests for CLI prediction functionality."""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection.predict import FraudDetectionPredictor, save_predictions

class TestCLIPrediction:
    """Test CLI prediction functionality."""
    
    def setup_method(self):
        """Setup test data and temporary files."""
        # Create test data
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
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = os.path.join(self.temp_dir, "test_data.csv")
        self.model_path = os.path.join(self.temp_dir, "test_model.pkl")
        self.output_path = os.path.join(self.temp_dir, "predictions.csv")
        
        # Save test data
        self.test_data.to_csv(self.test_data_path, index=False)
        
        # Create a dummy model for testing
        self._create_dummy_model()
    
    def teardown_method(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def _create_dummy_model(self):
        """Create a dummy model for testing."""
        from fraud_detection.model import FraudDetectionModel
        
        config = {
            'model': {
                'type': 'sklearn.LogisticRegression',
                'params': {
                    'max_iter': 100,
                    'class_weight': 'balanced'
                }
            }
        }
        
        model = FraudDetectionModel(config)
        
        # Prepare and train on test data
        from fraud_detection.data import prepare_features
        df_processed = prepare_features(self.test_data)
        X = df_processed.drop(columns=['isFraud'])
        y = df_processed['isFraud']
        
        model.train(X, y)
        model.save_model(self.model_path)
    
    def test_predictor_creation(self):
        """Test that predictor can be created."""
        predictor = FraudDetectionPredictor(self.model_path)
        assert predictor is not None
        assert predictor.model is not None
    
    def test_batch_prediction(self):
        """Test batch prediction functionality."""
        predictor = FraudDetectionPredictor(self.model_path)
        
        # Prepare test data
        from fraud_detection.data import prepare_features
        df_processed = prepare_features(self.test_data)
        X = df_processed.drop(columns=['isFraud'])
        
        # Make predictions
        predictions = predictor.predict_batch(X)
        
        # Check predictions
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_batch_prediction_probabilities(self):
        """Test batch prediction probabilities."""
        predictor = FraudDetectionPredictor(self.model_path)
        
        # Prepare test data
        from fraud_detection.data import prepare_features
        df_processed = prepare_features(self.test_data)
        X = df_processed.drop(columns=['isFraud'])
        
        # Get probabilities
        probabilities = predictor.predict_proba_batch(X)
        
        # Check probabilities
        assert probabilities.shape == (len(X), 2)  # Binary classification
        assert all(prob >= 0.0 and prob <= 1.0 for prob in probabilities.flatten())
        assert all(np.isclose(prob.sum(axis=1), 1.0))  # Probabilities sum to 1
    
    def test_single_prediction(self):
        """Test single transaction prediction."""
        predictor = FraudDetectionPredictor(self.model_path)
        
        # Test transaction data
        transaction_data = {
            'step': 1,
            'type': 'PAYMENT',
            'amount': 100.0,
            'nameOrig': 'C123',
            'oldbalanceOrg': 1000.0,
            'newbalanceOrig': 900.0,
            'nameDest': 'M123',
            'oldbalanceDest': 0.0,
            'newbalanceDest': 100.0,
            'isFraud': 0,
            'isFlaggedFraud': 0
        }
        
        result = predictor.predict_single(transaction_data)
        
        # Check result structure
        assert 'prediction' in result
        assert 'probability' in result
        assert 'is_fraud' in result
        
        # Check types
        assert isinstance(result['prediction'], int)
        assert isinstance(result['probability'], float)
        assert isinstance(result['is_fraud'], bool)
        
        # Check values
        assert result['prediction'] in [0, 1]
        assert 0.0 <= result['probability'] <= 1.0
        assert result['is_fraud'] == bool(result['prediction'])
    
    def test_save_predictions(self):
        """Test saving predictions to CSV."""
        # Create sample predictions
        predictions = np.array([0, 1, 0, 0, 1])
        
        # Save predictions
        save_predictions(predictions, self.output_path)
        
        # Check file exists
        assert os.path.exists(self.output_path)
        
        # Load and verify
        df_predictions = pd.read_csv(self.output_path)
        assert len(df_predictions) == len(predictions)
        assert 'prediction' in df_predictions.columns
        assert 'is_fraud' in df_predictions.columns
        
        # Check values
        assert all(df_predictions['prediction'] == predictions)
        assert all(df_predictions['is_fraud'] == (predictions == 1))
    
    def test_predictor_with_invalid_model(self):
        """Test predictor with invalid model path."""
        invalid_path = "nonexistent_model.pkl"
        
        with pytest.raises(Exception):
            FraudDetectionPredictor(invalid_path)
    
    def test_prediction_output_schema(self):
        """Test that prediction output has correct schema."""
        predictor = FraudDetectionPredictor(self.model_path)
        
        # Prepare test data
        from fraud_detection.data import prepare_features
        df_processed = prepare_features(self.test_data)
        X = df_processed.drop(columns=['isFraud'])
        
        # Make predictions
        predictions = predictor.predict_batch(X)
        
        # Save predictions
        save_predictions(predictions, self.output_path)
        
        # Load and check schema
        df_predictions = pd.read_csv(self.output_path)
        
        # Check columns
        expected_columns = ['prediction', 'is_fraud']
        assert all(col in df_predictions.columns for col in expected_columns)
        
        # Check data types
        assert df_predictions['prediction'].dtype in ['int64', 'int32']
        assert df_predictions['is_fraud'].dtype == 'bool'
        
        # Check value ranges
        assert all(df_predictions['prediction'].isin([0, 1]))
        assert all(df_predictions['is_fraud'].isin([True, False]))
