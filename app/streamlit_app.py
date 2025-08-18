"""Streamlit application for fraud detection."""

import streamlit as st
import pandas as pd
import yaml
import logging
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from fraud_detection.predict import FraudDetectionPredictor
from fraud_detection.utils import load_yaml_config, setup_logging

# Setup logging
logger = setup_logging()

def load_config() -> dict:
    """Load application configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "app.yaml"
    try:
        return load_yaml_config(str(config_path))
    except Exception as e:
        logger.warning(f"Could not load config: {e}. Using defaults.")
        return {
            "title": "Fraud Detection Demo",
            "sample_path": "data/sample.csv",
            "model_path": "artifacts/fraud_detection_pipeline.pkl"
        }

def load_model(config: dict):
    """Load the trained model."""
    model_path = config.get("model_path", "artifacts/fraud_detection_pipeline.pkl")
    
    if not Path(model_path).exists():
        st.error(f"Model not found at {model_path}. Please train the model first.")
        st.info("Run: poetry run fd-train --config configs/train.yaml")
        return None
    
    try:
        predictor = FraudDetectionPredictor(model_path)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def main():
    """Main Streamlit application."""
    # Load configuration
    config = load_config()
    
    # Set page config
    st.set_page_config(
        page_title=config.get("title", "Fraud Detection"),
        page_icon="🔍",
        layout="wide"
    )
    
    # Header
    st.title(config.get("title", "Fraud Detection Demo"))
    st.markdown("Enter transaction details below to detect potential fraud.")
    
    st.divider()
    
    # Load model
    predictor = load_model(config)
    if predictor is None:
        st.stop()
    
    # Input form
    with st.form("fraud_detection_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            transaction_type = st.selectbox(
                "Transaction Type",
                ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"],
                help="Type of financial transaction"
            )
            
            amount = st.number_input(
                "Amount",
                min_value=0.0,
                value=1000.0,
                step=100.0,
                help="Transaction amount in currency units"
            )
            
            oldbalanceOrg = st.number_input(
                "Old Balance (Sender)",
                min_value=0.0,
                value=10000.0,
                step=1000.0,
                help="Sender's balance before transaction"
            )
            
            newbalanceOrig = st.number_input(
                "New Balance (Sender)",
                min_value=0.0,
                value=9000.0,
                step=1000.0,
                help="Sender's balance after transaction"
            )
        
        with col2:
            oldbalanceDest = st.number_input(
                "Old Balance (Receiver)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                help="Receiver's balance before transaction"
            )
            
            newbalanceDest = st.number_input(
                "New Balance (Receiver)",
                min_value=0.0,
                value=0.0,
                step=1000.0,
                help="Receiver's balance after transaction"
            )
        
        # Submit button
        submitted = st.form_submit_button("Detect Fraud", type="primary")
        
        if submitted:
            # Prepare input data
            input_data = {
                "step": 1,
                "type": transaction_type,
                "amount": amount,
                "nameOrig": "C1234567890",
                "oldbalanceOrg": oldbalanceOrg,
                "newbalanceOrig": newbalanceOrig,
                "nameDest": "M9876543210",
                "oldbalanceDest": oldbalanceDest,
                "newbalanceDest": newbalanceDest,
                "isFraud": 0,  # Placeholder, will be predicted
                "isFlaggedFraud": 0
            }
            
            try:
                # Make prediction
                result = predictor.predict_single(input_data)
                
                # Display results
                st.subheader("🔍 Fraud Detection Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if result['is_fraud']:
                        st.error("🚨 **FRAUD DETECTED**")
                        st.metric("Risk Level", "HIGH", delta="⚠️")
                    else:
                        st.success("✅ **NO FRAUD DETECTED**")
                        st.metric("Risk Level", "LOW", delta="✅")
                
                with col2:
                    st.metric(
                        "Fraud Probability",
                        f"{result['probability']:.2%}",
                        delta=f"{result['probability']:.1%}"
                    )
                
                with col3:
                    st.metric(
                        "Prediction",
                        "Fraudulent" if result['is_fraud'] else "Legitimate",
                        delta="High Risk" if result['is_fraud'] else "Low Risk"
                    )
                
                # Additional insights
                st.subheader("📊 Transaction Analysis")
                
                # Balance changes
                balance_change_orig = oldbalanceOrg - newbalanceOrig
                balance_change_dest = newbalanceDest - oldbalanceDest
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Sender Balance Change",
                        f"${balance_change_orig:,.2f}",
                        delta=f"{balance_change_orig:,.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Receiver Balance Change",
                        f"${balance_change_dest:,.2f}",
                        delta=f"{balance_change_dest:,.2f}"
                    )
                
                # Risk factors
                risk_factors = []
                if amount > 10000:
                    risk_factors.append("High transaction amount")
                if balance_change_orig < 0:
                    risk_factors.append("Negative sender balance change")
                if transaction_type in ["TRANSFER", "CASH_OUT"]:
                    risk_factors.append("High-risk transaction type")
                
                if risk_factors:
                    st.warning("⚠️ **Risk Factors Identified:**")
                    for factor in risk_factors:
                        st.write(f"• {factor}")
                else:
                    st.info("ℹ️ **No significant risk factors identified**")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {e}")
                logger.error(f"Prediction error: {e}")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This fraud detection system uses machine learning to identify potentially fraudulent financial transactions.
        
        **Features analyzed:**
        - Transaction type and amount
        - Balance changes
        - Account patterns
        
        **Model:** Logistic Regression with balanced class weights
        """)
        
        st.header("📈 Model Performance")
        st.metric("Accuracy", "94%")
        st.metric("Fraud Recall", "94%")
        st.metric("Fraud Precision", "2%")
        
        st.info("""
        *Note: Low precision is due to severe class imbalance in the training data (0.04% fraud rate).
        The model prioritizes catching fraud cases over false positives.*
        """)
        
        st.header("🔧 Development")
        st.markdown("""
        Built with:
        - **Streamlit** for the web interface
        - **Scikit-learn** for ML pipeline
        - **Pydantic** for data validation
        - **Poetry** for dependency management
        """)

if __name__ == "__main__":
    main()
