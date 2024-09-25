import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def build_lstm_model(input_shape):
    """
    Build an LSTM model with dropout layers for better regularization.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=input_shape),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def scale_lstm_data(X_train, X_test):
    """
    Scale data between (-1, 1) for LSTM input.
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def reshape_lstm_data(X_train, X_test):
    """
    Reshape the input for LSTM with the correct batch size, timesteps, and features.
    """
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    return X_train, X_test

def evaluate_lstm_model(model, X_test, y_test):
    """
    Evaluate the LSTM model using RMSE.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    return rmse, predictions
