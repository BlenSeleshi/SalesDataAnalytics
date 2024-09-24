import statsmodels.api as sm
import matplotlib.pyplot as plt
import logging

def plot_acf_pacf(series, lags=40):
    logging.info("Plotting ACF and PACF.")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sm.graphics.tsa.plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title("Autocorrelation Function (ACF)")
    
    sm.graphics.tsa.plot_pacf(series.dropna(), lags=lags, ax=axes[1])
    axes[1].set_title("Partial Autocorrelation Function (PACF)")
    
    plt.show()
    logging.info("ACF and PACF plots generated.")