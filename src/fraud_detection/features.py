"""Feature engineering and preprocessing module for fraud detection."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging

logger = logging.getLogger(__name__)

class FeatureProcessor:
    """Feature processor for fraud detection data."""
    
    def __init__(self):
        self.preprocessor = None
        self.is_fitted = False
        
    def create_preprocessor(self, categorical_features: List[str], numerical_features: List[str]) -> ColumnTransformer:
        """Create the preprocessing pipeline."""
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        logger.info(f"Created preprocessor with {len(categorical_features)} categorical and {len(numerical_features)} numerical features")
        return preprocessor
    
    def fit_transform(self, X: pd.DataFrame, categorical_features: List[str], numerical_features: List[str]) -> np.ndarray:
        """Fit the preprocessor and transform the data."""
        if self.preprocessor is None:
            self.create_preprocessor(categorical_features, numerical_features)
        
        X_transformed = self.preprocessor.fit_transform(X)
        self.is_fitted = True
        
        logger.info(f"Fitted preprocessor and transformed data to shape {X_transformed.shape}")
        return X_transformed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        X_transformed = self.preprocessor.transform(X)
        logger.info(f"Transformed data to shape {X_transformed.shape}")
        return X_transformed
    
    def save_preprocessor(self, file_path: str):
        """Save the fitted preprocessor."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        joblib.dump(self.preprocessor, file_path)
        logger.info(f"Saved preprocessor to {file_path}")
    
    def load_preprocessor(self, file_path: str):
        """Load a saved preprocessor."""
        self.preprocessor = joblib.load(file_path)
        self.is_fitted = True
        logger.info(f"Loaded preprocessor from {file_path}")

def get_feature_names(categorical_features: List[str], numerical_features: List[str], X_sample: pd.DataFrame) -> List[str]:
    """Get feature names after preprocessing."""
    # Create a temporary preprocessor to get feature names
    temp_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    # Fit on sample data to get feature names
    temp_preprocessor.fit(X_sample)
    
    # Get feature names
    feature_names = []
    
    # Numerical features
    feature_names.extend(numerical_features)
    
    # Categorical features (after one-hot encoding)
    cat_encoder = temp_preprocessor.named_transformers_['cat']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    feature_names.extend(cat_feature_names)
    
    return feature_names

def create_feature_pipeline(categorical_features: List[str], numerical_features: List[str]) -> Pipeline:
    """Create a complete feature processing pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor)
    ])
    
    logger.info("Created feature processing pipeline")
    return pipeline
