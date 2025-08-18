# Fraud Detection ML

Machine learning model for detecting fraudulent transactions in financial transactions.

## Overview

This project implements an end-to-end machine learning pipeline for fraud detection in financial transactions. It includes data preprocessing, feature engineering, model training, and a Streamlit web application for real-time predictions.

## Features

- **Data Processing**: Automated data validation and preprocessing
- **Feature Engineering**: Balance difference calculations and transaction type encoding
- **Model Training**: Logistic Regression with balanced class weights
- **Web Interface**: Streamlit app for real-time fraud detection
- **CLI Tools**: Command-line interface for training and prediction
- **Type Safety**: Full type hints and validation with Pydantic
- **Testing**: Comprehensive test suite with pytest
- **CI/CD**: GitHub Actions for automated testing and validation

## Quick Start

### Prerequisites

- Python 3.12+
- Poetry for dependency management

### Installation

```bash
# Clone the repository
git clone https://github.com/junidebadii/machinelearning-fraud-detection.git
cd machinelearning-fraud-detection

# Install dependencies
poetry install

# Download data (optional)
poetry run python scripts/download_data.py

# Train the model
poetry run fd-train --config configs/train.yaml

# Make predictions
poetry run fd-predict --config configs/infer.yaml

# Run the Streamlit app
poetry run streamlit run app/streamlit_app.py
```

## Project Structure

```
fraud-detection-ml/
├─ src/fraud_detection/     # Core ML modules
├─ app/                     # Streamlit application
├─ configs/                 # Configuration files
├─ notebooks/               # Jupyter notebooks
├─ data/                    # Data files (sample only)
├─ artifacts/               # Model outputs (git-ignored)
├─ scripts/                 # Utility scripts
├─ tests/                   # Test suite
└─ docs/                    # Documentation
```

## Model Performance

- **Accuracy**: 94%
- **Precision**: 2% (fraud class)
- **Recall**: 94% (fraud class)
- **F1-Score**: 4% (fraud class)

*Note: Low precision is due to severe class imbalance (0.04% fraud rate)*

## Configuration

The project uses YAML configuration files for different components:

- `configs/train.yaml`: Training parameters and data paths
- `configs/infer.yaml`: Inference configuration
- `configs/app.yaml`: Streamlit app settings

## CLI Commands

- `fd-train`: Train the fraud detection model
- `fd-predict`: Make batch predictions
- `fd-app`: Launch the Streamlit application

## Development

### Code Quality

```bash
# Run linting
poetry run ruff check .

# Format code
poetry run black .

# Type checking
poetry run mypy .

# Run tests
poetry run pytest
```

### Pre-commit Hooks

```bash
poetry run pre-commit install
```

## Data

The dataset contains financial transaction records with the following features:
- Transaction type (PAYMENT, TRANSFER, CASH_OUT, DEPOSIT)
- Amount and balance information
- Fraud labels (target variable)

*Note: Full dataset not included due to size. Use `scripts/download_data.py` to fetch.*

Dataset available at: https://www.kaggle.com/datasets/amanalisiddiqui/fraud-detection-dataset?resource=download

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Model Card

See [docs/MODEL_CARD.md](docs/MODEL_CARD.md) for detailed model information.

## Data Card

See [docs/DATA_CARD.md](docs/DATA_CARD.md) for dataset documentation.
