import logging
import numpy as np
from sklearn.metrics import mean_squared_error

def evaluate_model_performance(y_true, y_pred, model_name):
    """
    Evaluates the model performance using RMSE and prints the results.
    """
    logging.info(f"Evaluating {model_name} performance.")
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    logging.info(f"{model_name} RMSE: {rmse}")
    return rmse

def compare_models(model1_performance, model2_performance, model1_name, model2_name):
    """
    Compares the performance of two models.
    """
    logging.info("Comparing model performance.")
    if model1_performance < model2_performance:
        logging.info(f"{model1_name} performs better with RMSE of {model1_performance}")
        return model1_name
    else:
        logging.info(f"{model2_name} performs better with RMSE of {model2_performance}")
        return model2_name
