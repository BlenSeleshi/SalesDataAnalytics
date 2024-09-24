import logging
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

def build_random_forest_model(X, y, preprocessor, param_grid):
    """
    Builds and trains a Random Forest model.
    """
    logging.info("Building Random Forest Model.")
    
    model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('model', RandomForestRegressor(random_state=42))])

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X, y)
    
    logging.info(f"Best parameters found: {grid_search.best_params_}")
    logging.info(f"Best score (MSE): {-grid_search.best_score_}")
    
    return grid_search.best_estimator_, grid_search.best_params_, -grid_search.best_score_

def predict_future_sales(model, future_data):
    """
    Predict future sales using the trained model.
    """
    logging.info("Predicting future sales.")
    predictions = model.predict(future_data)
    return predictions
