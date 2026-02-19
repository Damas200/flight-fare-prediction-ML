# Flight Fare Prediction Using Machine Learning

An end-to-end machine learning project that predicts airline ticket prices using historical flight data from Bangladesh. This project demonstrates a complete production-style ML workflow including data cleaning, feature engineering, model development, evaluation, and interpretability.

---

## Project Overview

Airlines and travel platforms rely on dynamic pricing strategies to optimize revenue and remain competitive. This project builds a regression model that predicts flight fares based on:

* Airline
* Route (Source & Destination)
* Aircraft Type
* Travel Class
* Seasonality
* Duration
* Days Before Departure

The objective is to estimate **Total Fare (BDT)** accurately and extract business insights from pricing patterns.

---

##  Problem Formulation

* **Task Type:** Supervised Regression
* **Target Variable:** `Total Fare (BDT)`
* **Dataset Size:** 57,000 records
* **Features:** 17 columns

---

##  Project Structure

```
flight_fare_project/
│
├── data/
│   └── Flight_Price_Dataset_of_Bangladesh.csv
│
├── models/
│   └── best_model.pkl
├── docs/
│   ├── problem_definition.md
│   └── project_report.md
│
├── data_loader.py
├── preprocessing.py
├── feature_engineering.py
├── modeling.py
├── evaluation.py
├── train_pipeline.py
│
├── Exploratory Data Analysis & Visualization.ipynb
├── requirements.txt
└── README.md
```

---

## ML Pipeline Architecture

```
Raw CSV
   ↓
Data Cleaning
   ↓
Feature Engineering
   ↓
Log Transformation
   ↓
Train/Test Split
   ↓
Model Training (Linear, Ridge, Lasso, Random Forest)
   ↓
Cross Validation
   ↓
Model Selection
   ↓
Evaluation & Interpretation
   ↓
Model Saved (.pkl)
```

All preprocessing and modeling steps are implemented using **Scikit-Learn Pipelines** to ensure reproducibility.

---

## Exploratory Data Analysis

Key insights:

* Flight fares are highly right-skewed (log transformation applied).
* Strong seasonal pricing effects (Hajj, Eid, Winter Holidays).
* Airline choice significantly influences average fare.
* Travel class and aircraft type are major pricing drivers.
* Days before departure has mild negative correlation with fare.

---

## Models Evaluated

| Model               | Test R²    | CV R²  | Overfitting |
| ------------------- | ---------- | ------ | ----------- |
| Linear Regression   | 0.8935     | 0.8929 | Very Low    |
| Ridge Regression    | 0.8935     | 0.8929 | Very Low    |
| Lasso Regression    | **0.8935** | 0.8929 | Very Low    |
| Tuned Random Forest | 0.8912     | 0.8902 | Moderate    |

---

## Final Model: Lasso Regression

Why Lasso?

* Strong predictive performance
* Minimal overfitting gap
* Feature regularization
* High interpretability

### Performance (Original Scale)

* **MAE:** ~28,576 BDT
* **RMSE:** ~48,448 BDT
* **Test R²:** 0.8935

Given an average fare of ~71,000 BDT, this represents strong predictive accuracy for real-world pricing data.

---

## Model Diagnostics

* Predicted vs Actual plot shows strong alignment
* Residual analysis indicates mild heteroscedasticity
* No evidence of severe overfitting
* Feature importance highlights business-relevant drivers

---

## Key Feature Drivers

Top predictive factors include:

* Aircraft Type
* Destination
* Travel Class
* Seasonality
* Duration
* Booking timing (Days Before Departure)

These insights align with real-world airline pricing behavior.

---

## How to Run This Project


### Create virtual environment

```bash
python3 -m venv flight_env
source flight_env/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run training pipeline

```bash
python train_pipeline.py
```

Model will be saved to:

```
models/best_model.pkl
```

---

## Requirements

Key libraries:

* pandas
* numpy
* scikit-learn
* matplotlib
* seaborn
* joblib

See `requirements.txt` for exact versions.

---

## Future Improvements

* Add Gradient Boosting (XGBoost / LightGBM)
* Integrate SHAP for advanced explainability
* Deploy using Streamlit or Flask
* Automate retraining via Airflow
* Include distance-based and external economic features

---

## Author

**Damas Niyonkuru**
 Data Engineer

---

