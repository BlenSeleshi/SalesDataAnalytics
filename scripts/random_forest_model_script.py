import logging
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

def build_random_forest_model(X_train, y_train, preprocessor, param_grid):
    """
    Builds a Random Forest Regressor using GridSearchCV and a pipeline.
    """
    logging.info("Building Random Forest Model.")
    
    # Define pipeline
    pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor(random_state=42))
    ])
    
    # Hyperparameter tuning
    search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
    search.fit(X_train, y_train)
    
    logging.info(f"Best parameters: {search.best_params_}")
    logging.info(f"Best score: {search.best_score_}")
    
    return search.best_estimator_, search.best_params_, search.best_score_

def serialize_model(model):
    """
    Serializes the model with a timestamp.
    """
    filename = f"model_{pd.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pkl"
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    logging.info(f"Model serialized as {filename}")
    return filename
