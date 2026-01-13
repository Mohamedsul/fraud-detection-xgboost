# Fraud Detection Using XGBoost

## 1. Executive Summary
This project is inspired by a real-world fraud prevention challenge faced daily by millions of consumers and businesses. Imagine a legitimate customer standing at a checkout counter while their card is unexpectedly declined — an experience that is frustrating, embarrassing, and disruptive. Behind this moment lies a machine learning system making a high‑stakes decision in milliseconds.

This project addresses that challenge by building an end-to-end fraud detection system using real-world e-commerce transaction data provided by Vesta Corporation as part of the IEEE-CIS Fraud Detection Kaggle competition. The goal is to accurately identify fraudulent transactions while minimizing false positives that negatively impact customer experience.

The solution follows a production-oriented data science workflow: exploratory data analysis, domain-driven feature engineering, XGBoost modeling, cost-sensitive threshold optimization, explainability using SHAP, and data drift monitoring. The emphasis is not only on predictive performance, but also on business impact, transparency, and long-term reliability.

---

## 2. Business Problem & Objectives

### 2.1 Industry Context
Fraud detection systems operate silently in the background of everyday transactions. While customers may only notice them when something goes wrong, these systems save consumers and businesses millions of dollars each year. However, overly aggressive fraud detection leads to false positives, damaging trust and customer experience.

The IEEE Computational Intelligence Society (IEEE-CIS), in partnership with Vesta Corporation — a leader in guaranteed e-commerce payment solutions — released a large-scale, real-world transaction dataset to benchmark and improve fraud detection models.

### 2.2 Business Problem
The core problem is to predict whether an online transaction is fraudulent (`isFraud = 1`) at the time of purchase using transaction-level, device-level, and product-related features.

### 2.3 Key Challenges
- Extreme class imbalance: fraud represents a very small fraction of total transactions
- Asymmetric error costs:
  - **False Negatives**: direct financial loss, reputational damage
  - **False Positives**: customer frustration, transaction abandonment
- Evolving fraud patterns and adversarial behavior

### 2.4 Project Objectives
- Build a high-performing fraud detection model evaluated using ROC-AUC
- Improve fraud detection accuracy without degrading customer experience
- Create interpretable and auditable predictions
- Design a system suitable for real-world deployment and monitoring

---

## 3. Data Understanding & Exploratory Data Analysis (EDA)

### 3.1 Data Overview
The dataset consists of transactional records with numerical, categorical, and time-based features, along with a binary target indicating fraud.

### 3.2 Class Imbalance Analysis
EDA revealed that fraudulent transactions account for a very small percentage of total transactions. This finding immediately ruled out accuracy as a reliable evaluation metric and motivated the use of recall, precision, and PR-AUC.

### 3.3 Behavioral Insights
Key patterns observed during EDA included:
- Fraudulent transactions tend to have higher or abnormal amounts
- Suspicious time-of-day and temporal patterns
- Rare or unusual categorical values
- Bursts of activity over short time windows

EDA was used to generate hypotheses about fraud behavior, which directly informed feature engineering decisions.

---

## 4. Feature Engineering

Feature engineering focused on capturing behavioral anomalies while avoiding data leakage.

### 4.1 Feature Categories
1. **Amount-Based Features**
   - Transaction amount relative to historical behavior
   - Normalized or ratio-based amount features

2. **Velocity & Frequency Features**
   - Number or sum of transactions within short time windows
   - Indicators of rapid transaction bursts

3. **Time-Based Features**
   - Hour of day, day of week
   - Weekend or off-hours indicators

4. **Categorical Encoding**
   - Encoding strategies chosen to preserve signal while minimizing overfitting

### 4.2 Data Leakage Prevention
All engineered features were validated to ensure availability at transaction time. Any feature relying on future information was excluded.

---

## 5. Modeling Approach

### 5.1 Model Selection
XGBoost was selected due to its:
- Strong performance on tabular data
- Ability to model non-linear relationships
- Robustness to missing values
- Compatibility with imbalanced datasets

### 5.2 Handling Class Imbalance
- Class weighting / scale_pos_weight applied
- Evaluation focused on recall, precision, and PR-AUC rather than accuracy

### 5.3 Model Training
The model was trained using a leakage-safe data split. Hyperparameters were selected to balance predictive power and generalization.

---

## 6. Threshold & Cost-Based Optimization

Rather than using the default 0.5 probability threshold, fraud detection was treated as a business decision problem.

### 6.1 Cost Framework
- False Negative cost: fraud loss + reputational impact
- False Positive cost: customer friction and operational overhead

### 6.2 Threshold Selection
Multiple thresholds were evaluated by analyzing confusion matrices and estimating total business cost. The final threshold was chosen to maximize fraud recall while keeping false positives within acceptable limits.

---

## 7. Model Explainability (SHAP)

Explainability is critical for trust, compliance, and operational use.

### 7.1 Global Explainability
SHAP feature importance was used to verify that the model relied on logical and stable fraud signals.

### 7.2 Local Explainability
Individual transaction explanations were generated to show why specific transactions were flagged as fraud.

### 7.3 Business Value
- Supports fraud analyst investigations
- Enables customer dispute handling
- Facilitates regulatory and audit requirements

---

## 8. Monitoring & Data Drift Detection

Fraud behavior evolves over time, making monitoring essential.

### 8.1 Drift Types
- **Data Drift**: Changes in input feature distributions
- **Concept Drift**: Changes in the relationship between features and fraud

### 8.2 Monitoring Strategy
- Feature distribution tracking
- Prediction score monitoring
- Drift metrics such as PSI

When drift is detected, the system can trigger retraining or threshold adjustments.

---

## 9. Results & Evaluation

### 9.1 Evaluation Metric
The competition evaluates models using **ROC-AUC**, which measures the model’s ability to rank fraudulent transactions higher than legitimate ones across all thresholds. This metric is well-suited for highly imbalanced datasets where accuracy is misleading.

### 9.2 Model Performance
The XGBoost model achieved strong ROC-AUC performance on validation data, demonstrating effective separation between fraudulent and non-fraudulent transactions.

Beyond ROC-AUC, additional metrics were analyzed to align with business objectives:
- **Recall (Fraud Detection Rate):** Ensures fraudulent transactions are identified
- **Precision:** Controls customer friction caused by false positives

### 9.3 Threshold-Based Results
After cost-based threshold tuning:
- Fraud recall increased significantly compared to the default 0.5 threshold
- False positives were kept within acceptable business limits
- Overall expected business cost was minimized

This confirms that decision threshold optimization is as important as model selection in fraud detection systems.

---

## 10. Production Considerations

In a real-world deployment, this system would include:
- Real-time scoring via an API
- Feature pipelines or feature store integration
- Logging of predictions, thresholds, and SHAP explanations
- Automated monitoring, alerting, and retraining workflows
- Champion/challenger model experimentation

---

## 11. Limitations & Future Work

### 11.1 Limitations
- The project relies on historical, static data and may not fully capture adversarial fraud evolution.
- Label noise is possible, as some fraudulent transactions may go undetected or mislabeled.
- ROC-AUC does not directly optimize business cost and must be complemented with threshold tuning.

### 11.2 Future Improvements
- Incorporate real-time feature pipelines and streaming data
- Add automated retraining based on drift detection
- Explore ensemble approaches or hybrid rules + ML systems
- Evaluate customer-level and merchant-level risk aggregation
- Integrate feedback loops from fraud analyst decisions

---

## 12. Conclusion

This project demonstrates a complete, production-oriented fraud detection system built on real-world e-commerce data. By combining robust modeling, cost-sensitive decision-making, explainability, and monitoring, the solution addresses both technical and business requirements.

The approach and structure presented in this report are directly transferable to real-world fraud, risk, and anomaly detection problems across financial and operational domains.

