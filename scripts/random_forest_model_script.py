import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np

def predict_future_sales(rf_model, X_test, num_weeks=6):
    """
    Predict 6 weeks (42 days) into the future using the trained Random Forest model.
    """
    logging.info(f"Predicting sales for the next {num_weeks} weeks.")
    
    predictions = []
    for i in range(num_weeks * 7):
        # Get the prediction for one day ahead
        next_pred = rf_model.predict(X_test)
        
        # Append predictions for analysis
        predictions.append(next_pred)
        
        # Slide the window forward by adding the latest prediction back to X_test (simulate roll-forward)
        X_test['Sales_lag_7'] = np.roll(X_test['Sales_lag_7'], -1)
        X_test['Sales_lag_7'][-1] = next_pred  # Use prediction for the next day
        
    return np.array(predictions).flatten()
