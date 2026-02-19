import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X_test, y_test):

    log_preds = model.predict(X_test)

    preds = np.expm1(log_preds)
    actual = np.expm1(y_test)

    mae = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))

    print("\nEvaluation Metrics (Original Scale):")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")

    return mae, rmse
