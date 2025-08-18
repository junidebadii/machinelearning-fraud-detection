"""Utility functions for fraud detection."""

import logging
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import os

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    # Create logger
    logger = logging.getLogger("fraud_detection")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # Add console handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise ValueError(f"Error loading config from {config_path}: {e}")

def save_yaml_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file."""
    try:
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    except Exception as e:
        raise ValueError(f"Error saving config to {config_path}: {e}")

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to {seed}")

def ensure_directory_exists(directory_path: str):
    """Ensure directory exists, create if it doesn't."""
    Path(directory_path).mkdir(parents=True, exist_ok=True)

def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent

def get_data_dir() -> Path:
    """Get the data directory path."""
    return get_project_root() / "data"

def get_artifacts_dir() -> Path:
    """Get the artifacts directory path."""
    return get_project_root() / "artifacts"

def get_configs_dir() -> Path:
    """Get the configs directory path."""
    return get_project_root() / "configs"

def validate_file_exists(file_path: str) -> bool:
    """Validate that a file exists."""
    return Path(file_path).exists()

def create_sample_data(input_file: str, output_file: str, n_samples: int = 100):
    """Create a sample dataset for testing."""
    import pandas as pd
    
    try:
        # Read original data
        df = pd.read_csv(input_file)
        
        # Take sample
        if len(df) > n_samples:
            df_sample = df.sample(n=n_samples, random_state=42)
        else:
            df_sample = df.copy()
        
        # Save sample
        ensure_directory_exists(Path(output_file).parent)
        df_sample.to_csv(output_file, index=False)
        
        print(f"Created sample dataset with {len(df_sample)} records at {output_file}")
        
    except Exception as e:
        print(f"Error creating sample data: {e}")
        raise
