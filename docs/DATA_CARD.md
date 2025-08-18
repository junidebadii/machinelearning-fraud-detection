# Fraud Detection Data Card

## Dataset Overview

**Dataset Name:** Financial Transaction Fraud Detection Dataset  
**Version:** 1.0.0  
**Date:** January 2025  
**Source:** Financial transaction records  
**License:** MIT License  
**Size:** 6,362,620 records  

## Dataset Description

### Purpose
This dataset is designed for training machine learning models to detect fraudulent financial transactions. It contains real-world transaction data with labeled fraud indicators.

### Scope
- **Domain:** Financial services and banking
- **Transaction Types:** PAYMENT, TRANSFER, CASH_OUT, DEPOSIT
- **Geographic Coverage:** Limited to specific financial systems
- **Time Period:** Historical transaction data

## Data Collection

### Collection Method
- **Source Systems:** Financial transaction processing systems
- **Collection Period:** Historical data collection
- **Update Frequency:** Static dataset (not real-time)
- **Data Quality:** Automated validation and cleaning

### Privacy and Ethics
- **Anonymization:** All personal identifiers are anonymized
- **Compliance:** Follows financial data privacy regulations
- **Consent:** Data collected under standard financial service agreements
- **PII Handling:** No personally identifiable information included

## Data Schema

### Feature Descriptions

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| `step` | Integer | Time step of transaction | 1 - N |
| `type` | Categorical | Transaction type | PAYMENT, TRANSFER, CASH_OUT, DEPOSIT |
| `amount` | Float | Transaction amount | 0.0 - ∞ |
| `nameOrig` | String | Origin account identifier | Anonymized strings |
| `oldbalanceOrg` | Float | Origin account balance before | 0.0 - ∞ |
| `newbalanceOrig` | Float | Origin account balance after | 0.0 - ∞ |
| `nameDest` | String | Destination account identifier | Anonymized strings |
| `oldbalanceDest` | Float | Destination account balance before | 0.0 - ∞ |
| `newbalanceDest` | Float | Destination account balance after | 0.0 - ∞ |
| `isFraud` | Integer | Fraud indicator (target) | 0 (legitimate), 1 (fraud) |
| `isFlaggedFraud` | Integer | System fraud flag | 0 (not flagged), 1 (flagged) |

### Data Types
- **Numerical:** step, amount, balance fields
- **Categorical:** type, nameOrig, nameDest
- **Binary:** isFraud, isFlaggedFraud

## Data Quality

### Completeness
- **Missing Values:** Minimal missing data
- **Coverage:** Complete transaction records
- **Data Integrity:** High integrity maintained

### Accuracy
- **Validation:** Automated validation rules
- **Consistency:** Consistent data formats
- **Reliability:** High reliability for training

### Bias Assessment
- **Class Imbalance:** Severe imbalance (0.04% fraud)
- **Transaction Distribution:** Varies by type
- **Geographic Bias:** Limited geographic coverage
- **Temporal Bias:** Historical patterns may not reflect current state

## Data Preprocessing

### Feature Engineering
- **Balance Differences:** Calculated from balance fields
- **Transaction Patterns:** Derived from type and amount
- **Normalization:** Standard scaling for numerical features

### Data Cleaning
- **Outlier Handling:** Statistical outlier detection
- **Format Standardization:** Consistent data formats
- **Validation Rules:** Schema validation and constraints

### Data Splits
- **Training Set:** 70% of data
- **Test Set:** 30% of data
- **Stratification:** Maintains fraud rate across splits

## Data Statistics

### Overall Statistics
- **Total Records:** 6,362,620
- **Features:** 11
- **Fraud Rate:** 0.04%
- **Legitimate Rate:** 99.96%

### Transaction Type Distribution
- **PAYMENT:** Most common transaction type
- **TRANSFER:** Second most common
- **CASH_OUT:** Less frequent
- **DEPOSIT:** Least frequent

### Amount Statistics
- **Mean Amount:** Varies by transaction type
- **Amount Range:** Wide range from small to large transactions
- **Distribution:** Right-skewed (many small, few large)

### Fraud Distribution
- **Fraud by Type:** Higher in TRANSFER and CASH_OUT
- **Fraud by Amount:** Varies by amount ranges
- **Fraud Patterns:** Complex patterns requiring ML detection

## Data Limitations

### Known Issues
- **Class Imbalance:** Very low fraud rate affects model training
- **Feature Dependencies:** Some features may be correlated
- **Temporal Patterns:** Limited temporal information
- **Geographic Coverage:** Limited to specific regions

### Data Gaps
- **Real-time Updates:** No real-time data streaming
- **Additional Context:** Limited external context information
- **Behavioral Patterns:** No user behavior history
- **Network Effects:** Limited transaction network analysis

## Usage Guidelines

### Recommended Uses
- **Model Training:** Fraud detection ML models
- **Research:** Financial fraud pattern analysis
- **Education:** ML and data science learning
- **Prototyping:** Fraud detection system development

### Not Recommended
- **Production Systems:** Without additional validation
- **Real-time Detection:** Due to historical nature
- **Regulatory Compliance:** Without proper validation
- **High-stakes Decisions:** Without human oversight

## Data Access

### Availability
- **Repository:** GitHub repository
- **Sample Data:** Included in data/sample.csv
- **Full Dataset:** Available via download script
- **Documentation:** Comprehensive usage guides

### Access Requirements
- **License:** MIT License
- **Attribution:** Credit to original source
- **Usage Rights:** Commercial and non-commercial use
- **Redistribution:** Allowed with license

## Maintenance

### Update Schedule
- **Version Updates:** As needed
- **Data Refresh:** Quarterly or as available
- **Quality Checks:** Continuous monitoring
- **Documentation:** Regular updates

### Support
- **Issues:** GitHub issue tracking
- **Documentation:** Comprehensive guides
- **Community:** Open source community support
- **Contact:** Maintainer contact information

## Citation

If you use this dataset in your work, please cite:

```
@dataset{fraud_detection_2025,
  title={Financial Transaction Fraud Detection Dataset},
  author={Junid Ebadi},
  year={2025},
  url={https://github.com/junidebadii/machinelearning-fraud-detection}
}
```

## Contact Information

**Dataset Maintainer:** Junid Ebadi  
**Email:** junid.ebadii@gmail.com  
**Repository:** https://github.com/junidebadii/machinelearning-fraud-detection  
**License:** MIT License
