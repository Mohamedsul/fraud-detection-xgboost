# 🛡️ Fraud Detection Using XGBoost (IEEE-CIS / Vesta)

## 📌 Project Overview
This project presents an **end-to-end fraud detection system** inspired by real-world e-commerce payment challenges.  
It is based on the **IEEE-CIS Fraud Detection** Kaggle competition, using transaction data provided by **Vesta Corporation**, a global leader in guaranteed e-commerce payments.

The goal is to accurately identify fraudulent transactions **while minimizing false positives** that negatively impact customer experience.

---

## 🧠 Business Problem
Fraud detection is a **cost-sensitive and highly imbalanced** classification problem:

- **False Negatives** → direct financial loss, reputational damage  
- **False Positives** → customer frustration, transaction abandonment  

This project prioritizes **business-aligned decision-making** over raw accuracy.

---

## ⚙️ Methodology
The project follows a production-oriented machine learning workflow:

1. Exploratory Data Analysis (EDA)
2. Domain-driven Feature Engineering
3. XGBoost Modeling
4. Cost-based Threshold Optimization
5. Model Explainability using SHAP
6. Data Drift Monitoring

---

## 📊 Evaluation
- **Primary Metric:** ROC-AUC (competition metric)
- **Business Metrics:** Recall & Precision
- **Decision Optimization:** Threshold tuned to minimize expected business cost rather than using default 0.5

---

## 🔍 Explainability & Trust
- SHAP used for **global feature importance** and **local transaction explanations**
- Supports analyst review, audits, and regulatory requirements

---

## 🔄 Production Readiness
- Drift detection for evolving fraud patterns
- Threshold tuning strategy
- Designed with real-time scoring pipelines in mind

---

## 📁 Repository Structure
```
fraud-detection-xgboost/
│
├── data/
│   ├── features_train.csv
│   ├── results.csv
│   ├── sample_submission.csv
│   ├── schema.json
│   ├── test_identity.csv
│   ├── test_transaction.csv
│   ├── train_identity.csv
│   └── train_transaction.csv
│
├── Model
│   └── xgb_model.pki
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_features.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_threshold_cost_tuning.ipynb
│   ├── 05_explainability_shap.ipynb
│   └── 06_drift_monitoring.ipynb
│
├── report/
│   └── Fraud_Detection_XGBoost_Report.pdf 
├── src/
│   ├── 1.config/
│       ├── model_params.yaml
│       └── settings.yaml
│   ├── 2.ingestion/
│       ├── load_transactions.py
│       └── schema_validation.py
│   ├── 3.features/
│       └── feature_pipeline.py
│   ├── 4.training/
│       └── train_xgboost.py
│   ├── 5.decisioning/ 
│       └── trhreshold_policy.py
│   ├── 6.explainability/
│       └── shap_service.py
│   ├── 7.monitoring/
│       └── data_drift.py
│   ├── 8.serving/
│       └── inference.py
│   └── 9.main/
│       └── main.py
│
└── README.md
```

---

## 🚀 Skills Demonstrated
- Fraud & risk modeling
- Imbalanced learning
- Cost-sensitive machine learning
- XGBoost
- SHAP explainability
- Production ML & monitoring mindset

---

## 📄 Full Project Report
📎 See the complete technical report here:  
➡️ `report/fraud_detection_with_xgboost_project_report.md`

---

## 🏷️ Keywords
Fraud Detection · XGBoost · Machine Learning · Risk Modeling · SHAP · Data Drift · FinTech
