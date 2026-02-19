#  Problem Definition & Data Understanding

## 1. Business Context

Airlines and travel platforms aim to estimate flight ticket prices accurately in order to:

- Support dynamic pricing strategies
- Improve revenue management
- Provide predictive pricing insights to customers
- Optimize seat allocation and demand forecasting

This project develops a machine learning regression model to predict flight fares in Bangladesh using structured flight and booking data.

---

## 2. Machine Learning Task

- Problem Type: Supervised Regression
- Target Variable: Total Fare (BDT)
- Objective: Predict total flight fare based on airline, route, timing, booking behavior, and seasonal information.
- Evaluation Metrics:
  - R² Score
  - Cross-Validation R²
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)

---

## 3. Dataset Overview

- Total Rows: 57,000
- Total Features: 17
- Target Column: Total Fare (BDT)

### Key Features Include:

- Airline
- Source & Destination
- Departure & Arrival Time
- Duration (hrs)
- Stopovers
- Aircraft Type
- Booking Source
- Seasonality
- Days Before Departure
- Base Fare
- Tax & Surcharge

---

## 4. Initial Observations

After loading the dataset:

- No structural corruption detected.
- Dataset size sufficient for regression modeling.
- Fare-related columns include potential target leakage.
- Date columns required proper preprocessing.
- Numerical and categorical features are mixed.

---

## 5. Target Leakage Handling

To ensure realistic prediction capability:

The following columns were removed from training:

- Base Fare (BDT)
- Tax & Surcharge (BDT)
- Total Fare (BDT)

These variables directly determine the target and would artificially inflate model performance if included.

---

## 6. Data Transformation Decisions

### Log Transformation

The target variable showed positive skewness.
To stabilize variance and improve regression performance:

```

y = log1p(Total Fare)

```

This reduces the influence of extreme values and improves model generalization.

---

## 7. Assumptions

- Historical pricing patterns remain stable.
- Feature relationships are consistent over time.
- No significant unseen airline or route shifts during prediction period.

---

## 8. Limitations

- No macroeconomic variables included.
- No demand elasticity indicators.
- No competitor pricing information.
- Static modeling (not time-series forecasting).

---

## 9. Summary

The dataset is well-structured and suitable for supervised regression modeling. 
After removing leakage and applying log transformation, it is ready for feature engineering and modeling.
```

