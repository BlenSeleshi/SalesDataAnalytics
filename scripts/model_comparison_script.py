import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

def evaluate_model_performance(actual_sales, predicted_sales, model_name="Model"):
    """
    Evaluates the performance of a model by calculating RMSE and MAE.
    """
    logging.info(f"Evaluating performance for {model_name}.")
    
    # Calculate RMSE and MAE
    rmse = np.sqrt(mean_squared_error(actual_sales, predicted_sales))
    mae = mean_absolute_error(actual_sales, predicted_sales)
    
    logging.info(f"{model_name} - RMSE: {rmse}, MAE: {mae}")
    
    return {'model': model_name, 'RMSE': rmse, 'MAE': mae}

def compare_models(models_performance):
    """
    Compare multiple models based on their RMSE and MAE.
    """
    logging.info("Comparing models' performance.")
    
    for performance in models_performance:
        model_name = performance['model']
        rmse = performance['RMSE']
        mae = performance['MAE']
        
        print(f"{model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}")
    
    best_model = min(models_performance, key=lambda x: x['RMSE'])
    logging.info(f"Best model is: {best_model['model']} with RMSE: {best_model['RMSE']}")
    
    return best_model
