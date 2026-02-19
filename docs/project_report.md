
#  Flight Fare Prediction Using Machine Learning

---

# Executive Summary

This project develops an end-to-end machine learning pipeline to predict airline ticket prices using historical flight data from Bangladesh. The objective is to assist airlines and travel platforms in estimating fares dynamically based on route, airline, seasonality, aircraft type, and booking timing.

Using 57,000 flight records, multiple regression models were trained and evaluated. After careful feature engineering, log transformation, cross-validation, and hyperparameter tuning, a **Lasso Regression model** achieved the best balance between performance and generalization, with:

* **Test R²:** 0.8935
* **MAE:** 28,576 BDT
* **RMSE:** 48,448 BDT

The model demonstrates strong predictive capability while maintaining low overfitting risk. Key drivers of fare variation include airline, aircraft type, destination, travel class, and seasonality.

---

# Business Problem

Airlines and travel platforms rely heavily on pricing strategies to maximize revenue while remaining competitive. Accurate fare estimation helps with:

* Dynamic pricing optimization
* Revenue management
* Demand forecasting
* Customer recommendation systems
* Seasonal pricing adjustments

The task was formulated as a **supervised regression problem**, where:

**Target Variable:**
`Total Fare (BDT)`

---

# Dataset Overview

* **Total observations:** 57,000
* **Total features:** 17
* **Data types:** Mix of categorical, numerical, and datetime
* **Missing values:** None
* **Memory usage:** ~7.4 MB

Key features include:

* Airline
* Source & Destination
* Duration
* Aircraft Type
* Class
* Seasonality
* Days Before Departure
* Base Fare & Tax components

---

# Exploratory Data Analysis (EDA)

## 1. Target Distribution

The original fare distribution was **highly right-skewed**, with values reaching 550,000 BDT.

After applying log transformation (`log1p`):

* Distribution became more symmetric
* Variance stabilized
* Model performance improved significantly

This justified modeling on the log-transformed target.

---

## 2. Airline Pricing Differences

Average fares vary by airline:

* Premium international carriers show higher average pricing
* Budget airlines show lower averages

This confirms airline identity as a strong pricing driver.

---

## 3. Seasonal Pricing Effects

Boxplots show significant variation across seasons:

* **Hajj season** → highest median and variability
* **Eid & Winter Holidays** → elevated pricing
* **Regular season** → lowest fares

Seasonality clearly impacts fare dynamics.

---

## 4. Correlation Analysis

Key findings:

* Base Fare and Tax are almost perfectly correlated with Total Fare (≈ 1.00)
* Duration shows moderate correlation (~0.35)
* Days Before Departure shows weak negative correlation

To prevent **target leakage**, Base Fare and Tax were excluded from modeling.

---

# Feature Engineering

The following transformations were applied:

* Datetime parsing
* Log transformation of target
* Removal of leakage features
* One-hot encoding for categorical variables
* Standard scaling for numerical variables
* Train/Test split (80/20, random_state=42)

All transformations were implemented using a **Scikit-Learn Pipeline**, ensuring reproducibility.

---

# Model Development & Comparison

Models evaluated:

* Linear Regression
* Ridge Regression
* Lasso Regression
* Tuned Random Forest

Evaluation strategy:

* Train/Test split
* 5-fold Cross Validation
* Overfitting gap analysis

### Results Summary

| Model               | Train R² | Test R² | CV R²  | Overfitting |
| ------------------- | -------- | ------- | ------ | ----------- |
| Linear Regression   | 0.8931   | 0.8935  | 0.8929 | Very Low    |
| Ridge               | 0.8931   | 0.8935  | 0.8929 | Very Low    |
| Lasso               | 0.8929   | 0.8935  | 0.8929 | Very Low    |
| Tuned Random Forest | 0.9295   | 0.8912  | 0.8902 | Moderate    |

---

## Final Model Selection: **Lasso Regression**

Why Lasso?

* Comparable performance to Linear Regression
* Better feature regularization
* More stable coefficients
* Minimal overfitting gap
* More interpretable

---

# Model Performance (Original Scale)

After reversing the log transformation:

* **MAE:** 28,576 BDT
* **RMSE:** 48,448 BDT

Given an average fare of ~71,000 BDT, this error range is reasonable for real-world dynamic pricing systems.

---

# Model Diagnostics

## Predicted vs Actual Plot

* Strong alignment in mid-range fares
* Slight underestimation of extremely high fares

## Residual Analysis

* Mild heteroscedasticity at higher price levels
* Acceptable model stability overall

---

# Feature Importance Insights

Top influential factors:

1. Aircraft Type
2. Destination
3. Travel Class
4. Seasonality
5. Days Before Departure

Business interpretation:

* Premium aircraft and long-haul routes increase pricing
* Travel class significantly impacts fare
* High-demand seasons raise prices
* Early booking slightly reduces fares

---

# Limitations

* Model may underestimate extreme luxury fares
* No external demand indicators included
* No real-time dynamic features (fuel price, competition pricing)
* Geographic distance not explicitly calculated

---

# Future Improvements

Potential enhancements:

* Include distance-based features
* Add holiday indicators
* Use Gradient Boosting (XGBoost / LightGBM)
* Deploy as REST API or Streamlit app
* Integrate with Airflow pipeline for automated retraining

---

# Conclusion

This project successfully demonstrates a complete machine learning workflow:

* Business framing
* Data cleaning & transformation
* Feature engineering
* EDA & visualization
* Model comparison & validation
* Interpretation & reporting
* Reproducible ML pipeline
* Model persistence

The final model achieves strong predictive performance with low overfitting and meaningful business insights.

