import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV
)

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def train_models(df, target_column, sample_size=None):
    """
    Train multiple regression models for flight fare prediction.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and feature-engineered dataset.
    target_column : str
        Name of target column.
    sample_size : int or None
        If provided, randomly samples dataset for faster training (dev mode).

    Returns
    -------
    results : dict
        Model performance dictionary.
    X_test : pd.DataFrame
    y_test : np.array
    """

    print("\n================ Model Training ================\n")

    # Development Mode Sampling
    # ==========================================
    if sample_size is not None and len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
        print(f"Development Mode: Using sample of {sample_size} rows.\n")


    # Remove Target Leakage
    # ==========================================
    leakage_columns = [
        "Base Fare (BDT)",
        "Tax & Surcharge (BDT)",
        "Total Fare (BDT)"
    ]

    # Target (log transformed)
    y = np.log1p(df[target_column])

    # Drop leakage columns safely
    X = df.drop(columns=[col for col in leakage_columns if col in df.columns])


    # Feature Type Separation
    # ==========================================
    categorical_features = X.select_dtypes(include=["object"]).columns

    # Exclude datetime explicitly
    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Preprocessing Pipeline
    # ==========================================
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Train/Test Split
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    # Linear Models
    # ==========================================
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=5000),
    }

    results = {}

    for name, model in models.items():

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)

        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)

        train_r2 = r2_score(y_train, train_preds)
        test_r2 = r2_score(y_test, test_preds)

        cv_scores = cross_val_score(
            pipeline,
            X,
            y,
            cv=5,
            scoring="r2",
            n_jobs=2  
        )

        results[name] = {
            "model": pipeline,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "cv_r2_mean": np.mean(cv_scores)
        }

        print(f"{name}")
        print(f"  Train R²: {train_r2:.4f}")
        print(f"  Test  R²: {test_r2:.4f}")
        print(f"  CV Mean R²: {np.mean(cv_scores):.4f}")
        print(f"  Overfitting Gap: {train_r2 - test_r2:.4f}")
        print("-" * 50)

    # Tuned Random Forest
    # ==========================================
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", RandomForestRegressor(
            random_state=42,
            n_jobs=2
        ))
    ])

    param_dist = {
        "model__n_estimators": [100, 150],
        "model__max_depth": [None, 15],
        "model__min_samples_split": [2, 5],
        "model__min_samples_leaf": [1, 2]
    }

    search = RandomizedSearchCV(
        rf_pipeline,
        param_distributions=param_dist,
        n_iter=6,
        cv=3,
        scoring="r2",
        n_jobs=2,
        random_state=42,
        verbose=1
    )

    search.fit(X_train, y_train)

    best_rf = search.best_estimator_

    train_preds = best_rf.predict(X_train)
    test_preds = best_rf.predict(X_test)

    train_r2 = r2_score(y_train, train_preds)
    test_r2 = r2_score(y_test, test_preds)

    cv_scores = cross_val_score(
        best_rf,
        X,
        y,
        cv=3,
        scoring="r2",
        n_jobs=2
    )

    results["Tuned Random Forest"] = {
        "model": best_rf,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "cv_r2_mean": np.mean(cv_scores)
    }

    print("Tuned Random Forest")
    print(f"  Train R²: {train_r2:.4f}")
    print(f"  Test  R²: {test_r2:.4f}")
    print(f"  CV Mean R²: {np.mean(cv_scores):.4f}")
    print(f"  Overfitting Gap: {train_r2 - test_r2:.4f}")
    print("-" * 50)

    print("\nModel training completed successfully.\n")

    return results, X_test, y_test
