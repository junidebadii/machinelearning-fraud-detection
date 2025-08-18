#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Running Fraud Detection ML Pipeline"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Check if Poetry is available
if ! command -v poetry &> /dev/null; then
    echo "❌ Error: Poetry not found. Please install Poetry first."
    exit 1
fi

echo "📋 Step 1: Code Quality Checks"
echo "-------------------------------"

echo "🔍 Running Ruff linting..."
poetry run ruff check . || { echo "❌ Linting failed"; exit 1; }
echo "✅ Linting passed"

echo "🎨 Running Black formatting check..."
poetry run black --check . || { echo "❌ Formatting check failed"; exit 1; }
echo "✅ Formatting check passed"

echo "🔍 Running MyPy type checking..."
poetry run mypy . || { echo "❌ Type checking failed"; exit 1; }
echo "✅ Type checking passed"

echo ""
echo "🧪 Step 2: Running Tests"
echo "-------------------------"

echo "🔬 Running pytest with coverage..."
poetry run pytest -q --cov=src --cov-report=term-missing || { echo "❌ Tests failed"; exit 1; }
echo "✅ Tests passed"

echo ""
echo "🤖 Step 3: Model Training"
echo "--------------------------"

echo "📊 Training fraud detection model..."
poetry run fd-train --config configs/train.yaml || { echo "❌ Training failed"; exit 1; }
echo "✅ Model training completed"

echo ""
echo "🔮 Step 4: Making Predictions"
echo "------------------------------"

echo "📈 Running inference on sample data..."
poetry run fd-predict --config configs/infer.yaml || { echo "❌ Inference failed"; exit 1; }
echo "✅ Inference completed"

echo ""
echo "🎉 All steps completed successfully!"
echo "======================================"
echo ""
echo "🚀 To run the Streamlit app:"
echo "   poetry run streamlit run app/streamlit_app.py"
echo ""
echo "📊 To view results:"
echo "   - Model: artifacts/fraud_detection_pipeline.pkl"
echo "   - Predictions: artifacts/preds.csv"
echo ""
echo "✨ Pipeline execution completed!"
