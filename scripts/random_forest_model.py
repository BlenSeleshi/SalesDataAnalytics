import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def build_random_forest_model(X_train, y_train):
    """
    Build a Random Forest model with optimized hyperparameters using RandomizedSearchCV.
    """
    model = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [10, 30, 50],
        'max_depth': [3, 5]
    }

    grid_search = RandomizedSearchCV(
        model, param_grid, n_iter=10, cv=3, scoring='neg_mean_squared_error', n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_random_forest_model(model, X_test, y_test):
    """
    Evaluate the Random Forest model using RMSE.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse, predictions
