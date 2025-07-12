# 🏡 Ames Housing ETL Pipeline

This project is part of my **EliteTechIntern** internship. It automates the **ETL (Extract, Transform, Load)** process for the Ames Housing dataset using Python and scikit-learn.

---

## 🚀 Project Description

A streamlined ETL pipeline that:
- Loads and cleans the **AmesHousing.csv** dataset
- Removes unnecessary columns like `PID`, `Alley`, etc.
- Handles missing values using appropriate strategies
- Encodes categorical features and scales numeric ones
- Outputs a clean, transformed dataset ready for ML modeling

---

## 🛠️ Tools & Libraries

- `pandas` – Data handling  
- `scikit-learn` – Pipelines, Imputation, Encoding, Scaling  
- `numpy` – Numerical operations  

---

## 🔁 Pipeline Breakdown

1. **Extract**: Load raw dataset from CSV
2. **Transform**:  
   - Impute missing values  
   - Scale numerical features  
   - One-hot encode categorical features  
   - Combine everything into a single DataFrame
3. **Load**: Save the transformed data as a new CSV file

---

## 📁 Files

- `AmesHousing.csv` – Original dataset  
- `AmesHousing_transformed.csv` – Cleaned and transformed dataset  
- `etl_pipeline.py` – Full ETL pipeline script  

---
