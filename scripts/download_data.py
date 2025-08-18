#!/usr/bin/env python3
"""Script to download fraud detection dataset."""

import os
import sys
from pathlib import Path
import pandas as pd
import logging

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection.utils import setup_logging, ensure_directory_exists

def setup_logging():
    """Setup logging for the script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def download_kaggle_dataset(dataset_name: str = "mlg-ulb/creditcardfraud", output_dir: str = "data"):
    """Download dataset from Kaggle."""
    try:
        import kaggle
        
        # Check if Kaggle credentials are available
        if not os.path.exists(os.path.expanduser('~/.kaggle/kaggle.json')):
            logger.warning("Kaggle credentials not found. Please set up Kaggle API credentials.")
            logger.info("Visit: https://github.com/Kaggle/kaggle-api#api-credentials")
            return False
        
        # Create output directory
        ensure_directory_exists(output_dir)
        
        # Download dataset
        logger.info(f"Downloading dataset: {dataset_name}")
        kaggle.api.dataset_download_files(dataset_name, path=output_dir, unzip=True)
        
        logger.info(f"Dataset downloaded successfully to {output_dir}")
        return True
        
    except ImportError:
        logger.error("Kaggle package not installed. Install with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        return False

def create_sample_from_existing(input_file: str, output_file: str, n_samples: int = 1000):
    """Create a sample dataset from existing data."""
    try:
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
        
        # Read data
        logger.info(f"Reading data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Create sample
        if len(df) > n_samples:
            df_sample = df.sample(n=n_samples, random_state=42)
        else:
            df_sample = df.copy()
        
        # Save sample
        ensure_directory_exists(Path(output_file).parent)
        df_sample.to_csv(output_file, index=False)
        
        logger.info(f"Created sample dataset with {len(df_sample)} records at {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating sample data: {e}")
        return False

def main():
    """Main function."""
    logger = setup_logging()
    
    # Check if we have the original dataset
    original_data = "AIML Dataset.csv"
    
    if os.path.exists(original_data):
        logger.info("Original dataset found. Creating sample for development.")
        
        # Create sample data
        success = create_sample_from_existing(
            original_data, 
            "data/sample.csv", 
            n_samples=1000
        )
        
        if success:
            logger.info("Sample dataset created successfully!")
            logger.info("You can now run: poetry run fd-train --config configs/train.yaml")
        else:
            logger.error("Failed to create sample dataset")
            sys.exit(1)
    
    else:
        logger.info("Original dataset not found. Attempting to download from Kaggle...")
        
        # Try to download from Kaggle
        success = download_kaggle_dataset()
        
        if success:
            logger.info("Dataset downloaded successfully!")
        else:
            logger.warning("Could not download dataset. Please manually place your dataset in the project root.")
            logger.info("Expected filename: 'AIML Dataset.csv'")
            sys.exit(1)

if __name__ == "__main__":
    main()
