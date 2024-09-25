import logging
import matplotlib.pyplot as plt
import statsmodels.api as sm

def plot_acf_pacf(series, lags=40):
    """
    Plots ACF and PACF for the given time series data.
    """
    logging.info("Plotting ACF and PACF.")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=axes[0])
    axes[0].set_title('Autocorrelation Function')
    
    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=axes[1])
    axes[1].set_title('Partial Autocorrelation Function')
    
    plt.tight_layout()
    plt.show()
    
    logging.info("ACF and PACF plotted.")
