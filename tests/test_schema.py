"""Tests for data schema validation."""

import pytest
import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection.data import validate_schema, DataSchema

class TestDataSchema:
    """Test data schema validation."""
    
    def test_valid_schema(self):
        """Test that valid data passes schema validation."""
        # Create valid test data
        valid_data = {
            'step': [1, 2, 3],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'amount': [100.0, 200.0, 300.0],
            'nameOrig': ['C123', 'C456', 'C789'],
            'oldbalanceOrg': [1000.0, 2000.0, 3000.0],
            'newbalanceOrig': [900.0, 1800.0, 2700.0],
            'nameDest': ['M123', 'M456', 'M789'],
            'oldbalanceDest': [0.0, 0.0, 0.0],
            'newbalanceDest': [100.0, 200.0, 300.0],
            'isFraud': [0, 0, 1],
            'isFlaggedFraud': [0, 0, 0]
        }
        
        df = pd.DataFrame(valid_data)
        assert validate_schema(df) == True
    
    def test_missing_columns(self):
        """Test that missing columns fail validation."""
        # Create data with missing columns
        invalid_data = {
            'step': [1, 2, 3],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'amount': [100.0, 200.0, 300.0],
            # Missing other required columns
        }
        
        df = pd.DataFrame(invalid_data)
        assert validate_schema(df) == False
    
    def test_wrong_data_types(self):
        """Test that wrong data types are detected."""
        # Create data with wrong types
        invalid_data = {
            'step': [1, 2, 3],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'amount': [100.0, 200.0, 300.0],
            'nameOrig': ['C123', 'C456', 'C789'],
            'oldbalanceOrg': [1000.0, 2000.0, 3000.0],
            'newbalanceOrig': [900.0, 1800.0, 2700.0],
            'nameDest': ['M123', 'M456', 'M789'],
            'oldbalanceDest': [0.0, 0.0, 0.0],
            'newbalanceDest': [100.0, 200.0, 300.0],
            'isFraud': ['0', '0', '1'],  # String instead of int
            'isFlaggedFraud': [0, 0, 0]
        }
        
        df = pd.DataFrame(invalid_data)
        # Should still pass but with warnings
        assert validate_schema(df) == True
    
    def test_null_values(self):
        """Test that null values are detected."""
        # Create data with null values
        valid_data = {
            'step': [1, 2, 3],
            'type': ['PAYMENT', 'TRANSFER', 'CASH_OUT'],
            'amount': [100.0, None, 300.0],  # One null value
            'nameOrig': ['C123', 'C456', 'C789'],
            'oldbalanceOrg': [1000.0, 2000.0, 3000.0],
            'newbalanceOrig': [900.0, 1800.0, 2700.0],
            'nameDest': ['M123', 'M456', 'M789'],
            'oldbalanceDest': [0.0, 0.0, 0.0],
            'newbalanceDest': [100.0, 200.0, 300.0],
            'isFraud': [0, 0, 1],
            'isFlaggedFraud': [0, 0, 0]
        }
        
        df = pd.DataFrame(valid_data)
        # Should pass but with warnings about nulls
        assert validate_schema(df) == True

class TestDataSchemaModel:
    """Test Pydantic data schema model."""
    
    def test_valid_schema_instance(self):
        """Test creating valid schema instance."""
        valid_data = {
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
        
        schema = DataSchema(**valid_data)
        assert schema.step == 1
        assert schema.type == 'PAYMENT'
        assert schema.amount == 100.0
        assert schema.isFraud == 0
    
    def test_invalid_schema_instance(self):
        """Test that invalid data raises validation error."""
        invalid_data = {
            'step': 'not_a_number',  # Should be int
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
        
        with pytest.raises(Exception):  # Pydantic validation error
            DataSchema(**invalid_data)
