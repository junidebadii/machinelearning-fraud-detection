"""Tests for feature engineering and preprocessing."""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection.features import FeatureProcessor, create_feature_pipeline, get_feature_names

class TestFeatureProcessor:
    """Test feature processor functionality."""
    
    def setup_method(self):
        """Setup test data."""
        self.test_data = pd.DataFrame({
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'amount': [100.0, 200.0, 300.0],
            'oldbalanceOrg': [1000.0, 2000.0, 3000.0],
            'newbalanceOrig': [900.0, 1800.0, 2700.0],
            'oldbalanceDest': [0.0, 0.0, 0.0],
            'newbalanceDest': [100.0, 200.0, 300.0],
            'balanceDiffOrig': [100.0, 200.0, 300.0],
            'balanceDiffDest': [100.0, 200.0, 300.0]
        })
        
        self.categorical_features = ['type']
        self.numerical_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'balanceDiffOrig', 'balanceDiffDest']
    
    def test_create_preprocessor(self):
        """Test preprocessor creation."""
        processor = FeatureProcessor()
        preprocessor = processor.create_preprocessor(self.categorical_features, self.numerical_features)
        
        assert preprocessor is not None
        assert len(preprocessor.transformers) == 2
        assert preprocessor.transformers[0][0] == 'num'
        assert preprocessor.transformers[1][0] == 'cat'
    
    def test_fit_transform(self):
        """Test fit and transform functionality."""
        processor = FeatureProcessor()
        X_transformed = processor.fit_transform(
            self.test_data, 
            self.categorical_features, 
            self.numerical_features
        )
        
        assert processor.is_fitted == True
        assert X_transformed.shape[0] == len(self.test_data)
        assert X_transformed.shape[1] > len(self.numerical_features)  # More features due to one-hot encoding
    
    def test_transform_without_fit(self):
        """Test that transform fails without fitting."""
        processor = FeatureProcessor()
        
        with pytest.raises(ValueError, match="Preprocessor must be fitted before transforming data"):
            processor.transform(self.test_data)
    
    def test_save_load_preprocessor(self, tmp_path):
        """Test saving and loading preprocessor."""
        processor = FeatureProcessor()
        processor.fit_transform(
            self.test_data, 
            self.categorical_features, 
            self.numerical_features
        )
        
        # Save preprocessor
        save_path = tmp_path / "preprocessor.pkl"
        processor.save_preprocessor(str(save_path))
        assert save_path.exists()
        
        # Load preprocessor
        new_processor = FeatureProcessor()
        new_processor.load_preprocessor(str(save_path))
        assert new_processor.is_fitted == True
        
        # Test that it can transform data
        X_transformed = new_processor.transform(self.test_data)
        assert X_transformed.shape[0] == len(self.test_data)

class TestFeaturePipeline:
    """Test feature pipeline creation."""
    
    def test_create_feature_pipeline(self):
        """Test feature pipeline creation."""
        categorical_features = ['type']
        numerical_features = ['amount', 'oldbalanceOrg']
        
        pipeline = create_feature_pipeline(categorical_features, numerical_features)
        
        assert pipeline is not None
        assert hasattr(pipeline, 'fit')
        assert hasattr(pipeline, 'transform')
    
    def test_get_feature_names(self):
        """Test getting feature names after preprocessing."""
        test_data = pd.DataFrame({
            'type': ['PAYMENT', 'TRANSFER'],
            'amount': [100.0, 200.0],
            'oldbalanceOrg': [1000.0, 2000.0]
        })
        
        categorical_features = ['type']
        numerical_features = ['amount', 'oldbalanceOrg']
        
        feature_names = get_feature_names(categorical_features, numerical_features, test_data)
        
        # Should have numerical features + encoded categorical features
        assert 'amount' in feature_names
        assert 'oldbalanceOrg' in feature_names
        # Should have encoded type features (e.g., type_TRANSFER)
        assert any('type_' in name for name in feature_names)
    
    def test_pipeline_shape_stability(self):
        """Test that pipeline maintains shape stability."""
        test_data_1 = pd.DataFrame({
            'type': ['PAYMENT', 'TRANSFER'],
            'amount': [100.0, 200.0],
            'oldbalanceOrg': [1000.0, 2000.0]
        })
        
        test_data_2 = pd.DataFrame({
            'type': ['PAYMENT', 'CASH_OUT'],
            'amount': [150.0, 250.0],
            'oldbalanceOrg': [1500.0, 2500.0]
        })
        
        categorical_features = ['type']
        numerical_features = ['amount', 'oldbalanceOrg']
        
        pipeline = create_feature_pipeline(categorical_features, numerical_features)
        
        # Fit on first dataset
        pipeline.fit(test_data_1)
        X1_transformed = pipeline.transform(test_data_1)
        
        # Transform second dataset
        X2_transformed = pipeline.transform(test_data_2)
        
        # Both should have same number of features
        assert X1_transformed.shape[1] == X2_transformed.shape[1]
        assert X1_transformed.shape[0] == len(test_data_1)
        assert X2_transformed.shape[0] == len(test_data_2)
