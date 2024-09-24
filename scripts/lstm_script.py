import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def build_lstm_model(input_shape):
    """
    Builds a simple LSTM model for time series forecasting.
    """
    logging.info("Building LSTM Model.")
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    logging.info("LSTM Model built successfully.")
    return model

def create_supervised_data(series, window):
    """
    Transforms time series data into supervised learning data.
    """
    logging.info("Creating supervised learning data.")
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:(i + window)])
        y.append(series[i + window])
    return np.array(X), np.array(y)

def predict_future_sales_lstm(model, input_data, n_steps):
    """
    Predict future sales using the trained LSTM model.
    """
    logging.info("Predicting future sales using LSTM.")
    predictions = []
    for _ in range(n_steps):
        pred = model.predict(input_data[np.newaxis, :, np.newaxis])
        predictions.append(pred[0][0])
        input_data = np.append(input_data[1:], pred[0][0])
    return np.array(predictions)
