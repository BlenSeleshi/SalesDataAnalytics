import logging
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for a given model.
    """
    logging.info("Plotting feature importance.")
    importance = model.named_steps['model'].feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importance[indices], align='center')
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
    plt.show()
    
    logging.info("Feature importance plotted.")

def estimate_confidence_interval(predictions, y_true, confidence=0.95):
    """
    Estimate the confidence interval of the predictions.
    """
    logging.info("Estimating confidence intervals.")
    errors = y_true - predictions
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    ci_upper = mean_error + 1.96 * std_error / np.sqrt(len(errors))
    ci_lower = mean_error - 1.96 * std_error / np.sqrt(len(errors))
    
    logging.info(f"95% confidence interval: [{ci_lower}, {ci_upper}]")
    return ci_lower, ci_upper

def visualize_predictions(predicted_sales, actual_sales, title='Sales Prediction'):
    """
    Visualizes predicted vs actual sales over time.
    """
    logging.info("Visualizing predicted vs actual sales.")
    
    plt.figure(figsize=(14, 7))
    
    plt.plot(actual_sales, color='blue', label='Actual Sales')
    plt.plot(predicted_sales, color='red', linestyle='--', label='Predicted Sales')
    
    plt.title(title)
    plt.xlabel('Days')
    plt.ylabel('Sales')
    plt.legend()
    plt.show()

    logging.info("Sales predictions visualization completed.")
