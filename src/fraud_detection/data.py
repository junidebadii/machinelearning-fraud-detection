"""Data loading and validation module for fraud detection."""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class DataSchema(BaseModel):
    """Schema for fraud detection data."""
    step: int = Field(..., description="Time step of transaction")
    type: str = Field(..., description="Transaction type")
    amount: float = Field(..., description="Transaction amount")
    nameOrig: str = Field(..., description="Origin account name")
    oldbalanceOrg: float = Field(..., description="Origin account old balance")
    newbalanceOrig: float = Field(..., description="Origin account new balance")
    nameDest: str = Field(..., description="Destination account name")
    oldbalanceDest: float = Field(..., description="Destination account old balance")
    newbalanceDest: float = Field(..., description="Destination account new balance")
    isFraud: int = Field(..., description="Fraud indicator (0/1)")
    isFlaggedFraud: int = Field(..., description="System flagged fraud (0/1)")

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise

def validate_schema(df: pd.DataFrame) -> bool:
    """Validate that dataframe matches expected schema."""
    expected_columns = [
        'step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg',
        'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest',
        'isFraud', 'isFlaggedFraud'
    ]
    
    if not all(col in df.columns for col in expected_columns):
        logger.error(f"Missing columns. Expected: {expected_columns}, Got: {list(df.columns)}")
        return False
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Found null values: {null_counts[null_counts > 0]}")
    
    # Check data types
    if df['isFraud'].dtype != 'int64':
        logger.warning("isFraud column should be int64")
    
    if df['amount'].dtype != 'float64':
        logger.warning("amount column should be float64")
    
    logger.info("Data schema validation passed")
    return True

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features for modeling."""
    df_processed = df.copy()
    
    # Create balance difference features
    df_processed['balanceDiffOrig'] = df_processed['oldbalanceOrg'] - df_processed['newbalanceOrig']
    df_processed['balanceDiffDest'] = df_processed['newbalanceDest'] - df_processed['oldbalanceDest']
    
    # Drop unnecessary columns
    columns_to_drop = ['step', 'nameOrig', 'nameDest', 'isFlaggedFraud']
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    logger.info(f"Prepared features. Final shape: {df_processed.shape}")
    return df_processed

def split_data(df: pd.DataFrame, target_col: str, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""
    from sklearn.model_selection import train_test_split
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Split data: X_train={X_train.shape}, X_test={X_test.shape}")
    return X_train, X_test, y_train, y_test
